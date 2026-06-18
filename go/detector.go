package mp

import (
	"github.com/kljensen/snowball"
)

// DetectionResult - результат проверки.
type DetectionResult struct {
	IsAllowed bool
	Reason    string
	Word      string
}

// Detector - основной детектор.
type Detector struct {
	normalizer     *TextNormalizer
	rootBlacklist  map[string]struct{}
	blackAC 			 *AhoCorasick
	whiteAC 			 *AhoCorasick
}

// NewDetector - инициализация детектора.
func NewDetector(config *Config) *Detector {
	normalizer := NewTextNormalizer(config)
	// Инициализируем списки.
	rootBlacklist := make(map[string]struct{})
	for _, word := range config.RootBlacklist {
		norm := normalizer.normalize(word)
		if norm != "" {
			rootBlacklist[norm] = struct{}{}
		}
	}
	// Инициализируем Ахо-Корасик.
	var blackWords []string
	for word := range config.Blacklist {
		norm := normalizer.normalize(word)
		if norm != "" {
			blackWords = append(blackWords, norm)
		}
	}

	var whiteWords []string
	for _, word := range config.Whitelist {
		norm := normalizer.normalize(word)
		if norm != "" {
			whiteWords = append(whiteWords, norm)
		}
	}

	blackAC := NewAhoCorasick(blackWords)
	whiteAC := NewAhoCorasick(whiteWords)

	return &Detector{
		normalizer:    normalizer,
		rootBlacklist: rootBlacklist,
		blackAC:       blackAC,
		whiteAC:       whiteAC,
	}
}

// Check - основная функция, проверяет никнейм.
func (d *Detector) Check(nickname string) *DetectionResult {
	// Нормализуем ник (убираем лишние символы).
	normalized := d.normalizer.normalize(nickname)

	// Здесь мы находим все вхождения плохих слов в наш ник.
	badMatches := d.blackAC.findAll(normalized)

	if len(badMatches) > 0 {
		// Здесь находим вхождение слов из белого списка.
		whiteMatches := d.whiteAC.findAll(normalized)
		// если таких слов нет — сразу бан (нашли плохие).
		if len(whiteMatches) == 0 {
			return &DetectionResult{
				IsAllowed: false,
				Reason:    "substring_match",
				Word:      badMatches[0].Word,
			}
		}
		// Если есть, то нам нужно смотреть, входит ли плохое в состав хорошего.
		// Потому что человек может одновременно дать нам и плохое, и хорошее слово.
		// И если наше плохое покрывается хорошим, то мы пропускаем.
		// Если нет (то есть нашли плохое вне хорошего), то бан.
		for _, bad := range badMatches {
			if !isCovered(bad, whiteMatches) {
				return &DetectionResult{
					IsAllowed: false,
					Reason:    "substring_match",
					Word:      bad.Word,
				}
			}
		}
		// Если слово прошло через эту адскую проверку, то дальше не проверяем, approved.
		return &DetectionResult{IsAllowed: true}
	}
	// Стемминг. Так же Ахо-Корасиком ищем плохие корни.
	stemmed := stem(normalized)
	if stemmed != normalized {
		if matches := d.blackAC.findAll(stemmed); len(matches) > 0 {
			return &DetectionResult{
				IsAllowed: false,
				Reason:    "stem_match",
				Word:      matches[0].Word,
			}
		}
	}
	// Root-проверка.
	if _, exists := d.rootBlacklist[stemmed]; exists {
		return &DetectionResult{
			IsAllowed: false,
			Reason:    "forbidden_root",
			Word:      stemmed,
		}
	}

	return &DetectionResult{IsAllowed: true}
}

// Проверка на покрытие плохого слова хорошим.
func isCovered(bad Match, whites []Match) bool {
	for _, w := range whites {
		if bad.Start >= w.Start && bad.End <= w.End {
			return true
		}
	}
	return false
}

func stem(word string) string {
	stem, err := snowball.Stem(word, "russian", true)
	if err != nil {
		return word
	}
	return stem
}