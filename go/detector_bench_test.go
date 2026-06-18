package mp

import (
	"testing"
)

var benchConfig *Config
var benchDetector *Detector

func init() {
	var err error
	benchConfig, err = loadConfigFromFile("../config.json")
	if err != nil {
		panic(err)
	}
	benchDetector = NewDetector(benchConfig)
}

// Бенчмарк для нормальных ников (должны проходить).
func BenchmarkCheckNormalNick(b *testing.B) {
	nicks := []string{
		"молодец",
		"умник",
		"дуракоподобный",
		"притупился",
		"кот",
		"илья",
		"крысиный",
	}
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchDetector.Check(nicks[i%len(nicks)])
	}
}

// Бенчмарк для плохих ников (должны блокироваться).
func BenchmarkCheckBadNick(b *testing.B) {
	nicks := []string{
		"дурак",
		"Dypak",
		"дypaк",
		"дyp4к",
		"тупица",
		"Tuпitsa",
		"тупой",
		"дура",
		"бестолковый",
		"крыса",
		"крысовый",
		"дурак крысиный",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchDetector.Check(nicks[i%len(nicks)])
	}
}

// Бенчмарк для коротких ников.
func BenchmarkCheckShortNick(b *testing.B) {
	nicks := []string{
		"a",
		"ab",
		"abc",
		"123",
		"",
		" ",
		"x",
		"y",
		"z",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, nick := range nicks {
			benchDetector.Check(nick)
		}
	}
}

// Бенчмарк для ников с заменой символов.
func BenchmarkCheckObfuscatedNick(b *testing.B) {
	nicks := []string{
		"дyp4к",
		"дypaк",
		"Dypak",
		"Tuпitsa",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, nick := range nicks {
			benchDetector.Check(nick)
		}
	}
}

// Бенчмарк для длинных ников.
func BenchmarkCheckLongNick(b *testing.B) {
	longNick := "этооченьдлинныйниккоторыйдолженпроверитьсянадлительностьиневызватьпроблемспроизводительностью"
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchDetector.Check(longNick)
	}
}

// Бенчмарк для смешанных ников.
func BenchmarkCheckMixedNick(b *testing.B) {
	nicks := []string{
		"Dypakдурак",
		"Tuпitsaтупица",
		"крысаkrysa",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, nick := range nicks {
			benchDetector.Check(nick)
		}
	}
}

// Параллельный бенчмарк.
func BenchmarkCheckParallel(b *testing.B) {
	nick := "подозрительныйниккоторыйпроверяется"
	
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			benchDetector.Check(nick)
		}
	})
}