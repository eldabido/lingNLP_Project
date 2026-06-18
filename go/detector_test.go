package mp

import (
	"encoding/csv"
	"fmt"
	"os"
	"testing"
)

type testCase struct {
	nick     string
	expected bool   // true = разрешён, false = запрещён.
	category string // для группировки в отчёте.
}

func getAllTestCases() []testCase {
	return []testCase{
		// Здесь тесты.
	}
}

func TestDetector(t *testing.T) {
	config, err := loadConfigFromFile("../config.json")
	if err != nil {
		t.Fatalf("Не удалось загрузить config.json: %v", err)
	}
	detector := NewDetector(config)
	tests := getAllTestCases()

	// Счётчики для метрик.
	var tp, tn, fp, fn int           // общие
	catCorrect := map[string]int{}   // верных по категории
	catTotal := map[string]int{}     // всего по категории

	var errors []string // список ошибок для отчёта

	for _, tt := range tests {
		result := detector.Check(tt.nick)
		got := result.IsAllowed
		catTotal[tt.category]++

		if got == tt.expected {
			catCorrect[tt.category]++
			if tt.expected {
				tn++ // правильно разрешили (true negative для «запрещён»)
			} else {
				tp++ // правильно заблокировали
			}
		} else {
			if tt.expected {
				fp++ // ложное срабатывание (заблокировали хорошее)
			} else {
				fn++ // пропустили плохое
			}
			mark := "FN (пропущено)"
			if tt.expected {
				mark = "FP (ложное)"
			}
			reason := ""
			if !result.IsAllowed {
				reason = fmt.Sprintf(" [reason=%s, word=%s]", result.Reason, result.Word)
			}
			errors = append(errors, fmt.Sprintf(
				"  ✗ %-35s  ожидалось=%-10v  получено=%-10v  %s%s",
				"\""+tt.nick+"\"", tt.expected, got, mark, reason,
			))
		}
	}

	// ── Вывод результатов ──
	total := len(tests)
	correct := tp + tn
	
	fmt.Println()
	fmt.Println("  РЕЗУЛЬТАТЫ Go rule-based детектора")
	fmt.Printf("  Всего тестов:     %d\n", total)
	fmt.Printf("  Верных:           %d\n", correct)
	fmt.Printf("  Ошибок:           %d\n", total-correct)
	fmt.Println()

	// Accuracy.
	accuracy := float64(correct) / float64(total)
	fmt.Printf("  Accuracy:         %.3f\n", accuracy)

	// Precision, Recall, F1 — macro-averaged (среднее по обоим классам).
	// Класс 0 (blocked).
	var p0, r0, f0 float64
	if tp+fp > 0 {
		p0 = float64(tp) / float64(tp+fp)
	}
	if tp+fn > 0 {
		r0 = float64(tp) / float64(tp+fn)
	}
	if p0+r0 > 0 {
		f0 = 2 * p0 * r0 / (p0 + r0)
	}
	// Класс 1 (allowed).
	var p1, r1, f1c1 float64
	if tn+fn > 0 {
		p1 = float64(tn) / float64(tn+fn)
	}
	if tn+fp > 0 {
		r1 = float64(tn) / float64(tn+fp)
	}
	if p1+r1 > 0 {
		f1c1 = 2 * p1 * r1 / (p1 + r1)
	}
	// Macro.
	precision := (p0 + p1) / 2
	recall := (r0 + r1) / 2
	f1 := (f0 + f1c1) / 2

	fmt.Printf("  Precision (macro): %.3f\n", precision)
	fmt.Printf("  Recall (macro):    %.3f\n", recall)
	fmt.Printf("  F1 (macro):        %.3f\n", f1)
	fmt.Println()
	fmt.Printf("  TP=%d (верно заблокировано)  FP=%d (ложное срабатывание)\n", tp, fp)
	fmt.Printf("  FN=%d (пропущено)            TN=%d (верно разрешено)\n", fn, tn)
	fmt.Println()

	// По категориям.
	fmt.Println("  По категориям:")
	categories := []string{"blacklist", "obfuscated", "outside_blacklist", "unmapped_digits", "positive"}
	categoryLabels := map[string]string{
		"blacklist":         "Слова из blacklist     ",
		"obfuscated":        "Обфусцированные        ",
		"outside_blacklist":  "Вне blacklist (новые)  ",
		"unmapped_digits":   "Незамапленные цифры    ",
		"positive":          "Разрешённые (позитив)  ",
	}
	for _, cat := range categories {
		total := catTotal[cat]
		correct := catCorrect[cat]
		if total > 0 {
			pct := float64(correct) / float64(total) * 100
			label := categoryLabels[cat]
			fmt.Printf("    %s %2d/%2d  (%.0f%%)\n", label, correct, total, pct)
		}
	}

	// Ошибки.
	if len(errors) > 0 {
		fmt.Println()
		fmt.Printf("  Детали ошибок (%d):\n", len(errors))
		for _, e := range errors {
			fmt.Println(e)
		}
	}

	fmt.Println()

	// Фейлим тест только если есть ошибки на ОРИГИНАЛЬНЫХ кейсах или false positives.
	originalErrors := 0
	for _, tt := range tests {
		if tt.category == "outside_blacklist" || tt.category == "unmapped_digits" {
			continue // пропускаем ожидаемые ошибки
		}
		result := detector.Check(tt.nick)
		if result.IsAllowed != tt.expected {
			originalErrors++
			t.Errorf("nick: %q  ожидалось: %v  получено: %v", tt.nick, tt.expected, result.IsAllowed)
		}
	}
	if originalErrors == 0 {
		fmt.Println("Оригинальные тесты пройдены (ошибки только на кейсах вне blacklist)")
	}
}

func TestDetectorCSV(t *testing.T) {
	config, err := loadConfigFromFile("../config.json")
	if err != nil {
		t.Fatalf("Не удалось загрузить config.json: %v", err)
	}
	detector := NewDetector(config)
	tests := getAllTestCases()

	os.MkdirAll("../results", 0o755)

	predPath := "../results/go_predictions.csv"
	predFile, err := os.Create(predPath)
	if err != nil {
		t.Fatalf("Не удалось создать %s: %v", predPath, err)
	}
	defer predFile.Close()
	predFile.WriteString("\xEF\xBB\xBF")

	predWriter := csv.NewWriter(predFile)
	defer predWriter.Flush()
	predWriter.Write([]string{"nick", "expected", "got", "reason", "word", "category"})

	// Счётчики.
	type counters struct {
		tp, tn, fp, fn int
	}
	var allC, obfC counters

	for _, tt := range tests {
		result := detector.Check(tt.nick)
		got := result.IsAllowed

		// Записываем предсказание.
		expectedStr := "false"
		if tt.expected {
			expectedStr = "true"
		}
		gotStr := "false"
		if got {
			gotStr = "true"
		}
		predWriter.Write([]string{
			tt.nick, expectedStr, gotStr,
			result.Reason, result.Word, tt.category,
		})

		// Считаем в общие счётчики.
		c := &allC
		isObf := tt.category == "obfuscated" || tt.category == "unmapped_digits"

		if got == tt.expected {
			if tt.expected {
				c.tn++
				if isObf { obfC.tn++ }
			} else {
				c.tp++
				if isObf { obfC.tp++ }
			}
		} else {
			if tt.expected {
				c.fp++
				if isObf { obfC.fp++ }
			} else {
				c.fn++
				if isObf { obfC.fn++ }
			}
		}
	}

	macroMetrics := func(c counters) (accuracy, precision, recall, f1 float64) {
		total := c.tp + c.tn + c.fp + c.fn
		if total == 0 {
			return
		}
		accuracy = float64(c.tp+c.tn) / float64(total)

		var p0, r0, f0 float64
		if c.tp+c.fp > 0 { p0 = float64(c.tp) / float64(c.tp+c.fp) }
		if c.tp+c.fn > 0 { r0 = float64(c.tp) / float64(c.tp+c.fn) }
		if p0+r0 > 0 { f0 = 2 * p0 * r0 / (p0 + r0) }

		var p1, r1, f1c1 float64
		if c.tn+c.fn > 0 { p1 = float64(c.tn) / float64(c.tn+c.fn) }
		if c.tn+c.fp > 0 { r1 = float64(c.tn) / float64(c.tn+c.fp) }
		if p1+r1 > 0 { f1c1 = 2 * p1 * r1 / (p1 + r1) }

		precision = (p0 + p1) / 2
		recall = (r0 + r1) / 2
		f1 = (f0 + f1c1) / 2
		return
	}

	accAll, precAll, recAll, f1All := macroMetrics(allC)
	accObf, precObf, recObf, f1Obf := macroMetrics(obfC)

	metPath := "../results/go_metrics.csv"
	metFile, err := os.Create(metPath)
	if err != nil {
		t.Fatalf("Не удалось создать %s: %v", metPath, err)
	}
	defer metFile.Close()
	metFile.WriteString("\xEF\xBB\xBF")

	metWriter := csv.NewWriter(metFile)
	defer metWriter.Flush()
	metWriter.Write([]string{"subset", "accuracy", "precision", "recall", "f1"})
	metWriter.Write([]string{"all",
		fmt.Sprintf("%.4f", accAll), fmt.Sprintf("%.4f", precAll),
		fmt.Sprintf("%.4f", recAll), fmt.Sprintf("%.4f", f1All),
	})
	metWriter.Write([]string{"obfuscated",
		fmt.Sprintf("%.4f", accObf), fmt.Sprintf("%.4f", precObf),
		fmt.Sprintf("%.4f", recObf), fmt.Sprintf("%.4f", f1Obf),
	})

	fmt.Printf("\nПредсказания: %s (%d записей)\n", predPath, len(tests))
	fmt.Printf("Метрики:      %s\n", metPath)
	fmt.Printf("all:         Acc=%.3f  P=%.3f  R=%.3f  F1=%.3f\n", accAll, precAll, recAll, f1All)
	fmt.Printf("obfuscated:  Acc=%.3f  P=%.3f  R=%.3f  F1=%.3f\n", accObf, precObf, recObf, f1Obf)
}
