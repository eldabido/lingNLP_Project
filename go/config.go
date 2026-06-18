package mp

import (
	"encoding/json"
	"os"
)

// Config структура конфигурации.
type Config struct {
	// Правила замены символов (цифры/символы на буквы).
	CharReplacements map[string]string `json:"char_replacements"`

	// Транслитерация.
	TranslitMap 		 map[string][]string `json:"translit_map"`

	// Длинная транслитерация.
	LongPatterns     map[string]string `json:"long_patterns"`

	// Черный список слов.
	Blacklist 			 map[string]bool `json:"blacklist"`

	// Черный список корней.
	RootBlacklist    []string `json:"root_blacklist"`
	
	// Белый список слов.
	Whitelist 			 []string `json:"whitelist"`
}

// loadConfigFromFile загружает конфигурацию из JSON файла.
func loadConfigFromFile(path string) (*Config, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var config Config
	decoder := json.NewDecoder(file)
	err = decoder.Decode(&config)
	if err != nil {
		return nil, err
	}

	return &config, nil
}
