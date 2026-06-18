package mp

type node struct {
	children map[rune]*node
	fail     *node
	output   []string
}

type AhoCorasick struct {
	root *node
}

// Match - структура для вхождений. 
type Match struct {
	Start int
	End   int
	Word  string
}

func NewAhoCorasick(words []string) *AhoCorasick {
	root := &node{
		children: make(map[rune]*node),
	}

	// Строим trie.
	for _, word := range words {
		current := root
		for _, r := range word {
			if current.children[r] == nil {
				current.children[r] = &node{
					children: make(map[rune]*node),
				}
			}
			current = current.children[r]
		}
		current.output = append(current.output, word)
	}

	// Строим fail-ссылки (BFS).
	queue := []*node{}

	for _, child := range root.children {
		child.fail = root
		queue = append(queue, child)
	}

	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]

		for char, child := range current.children {
			failNode := current.fail

			for failNode != nil && failNode.children[char] == nil {
				failNode = failNode.fail
			}

			if failNode == nil {
				child.fail = root
			} else {
				child.fail = failNode.children[char]
				child.output = append(child.output, child.fail.output...)
			}

			queue = append(queue, child)
		}
	}

	return &AhoCorasick{root: root}
}

func (ac *AhoCorasick) findAll(text string) []Match {
	var matches []Match

	current := ac.root
	runes := []rune(text)

	for i, r := range runes {
		for current != nil && current.children[r] == nil {
			current = current.fail
		}

		if current == nil {
			current = ac.root
			continue
		}

		current = current.children[r]

		for _, word := range current.output {
			length := len([]rune(word))
			matches = append(matches, Match{
				Start: i - length + 1,
				End:   i + 1,
				Word:  word,
			})
		}
	}

	return matches
}
