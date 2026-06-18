package mp

import (
	"strings"
	"unicode"
)

const minLet = 2

// TextNormalizer - здесь мапы для нормализации.
type TextNormalizer struct {
	charReplacements map[string]string
	homoglyphs       map[rune]rune
	translitMap      map[string][]string
	longPatterns     map[string]string
}

// NewTextNormalizer - инициализация нормализатора.
func NewTextNormalizer(config *Config) *TextNormalizer {
	return &TextNormalizer{
		charReplacements: config.CharReplacements,
		translitMap: config.TranslitMap,
		longPatterns:     config.LongPatterns,
		homoglyphs: map[rune]rune{
			'a': 'а',
			'e': 'е',
			'o': 'о',
			'p': 'р',
			'c': 'с',
			'x': 'х',
			'y': 'у',
			'k': 'к',
			'b': 'б',
			'm': 'м',
			'h': 'н',
			'i': 'и',
			'l': 'л',
		},
	}
}

// normalize - нормализация ника.
func (n *TextNormalizer) normalize(text string) string {
    if text == "" {
        return ""
    }
    text = strings.ToLower(text)

    // Замена цифр и спецсимволов.
    for from, to := range n.charReplacements {
        text = strings.ReplaceAll(text, from, to)
    }

    // Гомоглифы.
    var builder strings.Builder
    for _, r := range text {
        if repl, ok := n.homoglyphs[r]; ok {
            builder.WriteRune(repl)
        } else {
            builder.WriteRune(r)
        }
    }
    text = builder.String()

    // Транслитерация.
    text = n.latToCyr(text)

    // Оставляем только буквы.
    clean := make([]rune, 0)
    for _, r := range text {
        if unicode.IsLetter(r) {
            clean = append(clean, r)
        }
    }

    return collapseRepeats(string(clean))
}

// latToCyr - замена в строке на русские символы.
func (n *TextNormalizer) latToCyr(text string) string {
	result := text
	for lat, cyr := range n.longPatterns {
		result = strings.ReplaceAll(result, lat, cyr)
	}
	for cyr, latVariants := range n.translitMap {
		for _, lat := range latVariants {
			result = strings.ReplaceAll(result, lat, cyr)
		}
	}
	return result
}

// collapseRepeats - убирает из строки повторения символов (Пример: ааааа).
func collapseRepeats(s string) string {
	var result []rune
	var prev rune
	count := 0

	for _, r := range s {
		if r == prev {
			count++
			if count < minLet {
				result = append(result, r)
			}
		} else {
			prev = r
			count = 1
			result = append(result, r)
		}
	}
	return string(result)
}
