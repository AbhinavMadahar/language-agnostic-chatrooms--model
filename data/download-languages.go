// downloads all the pairs for the languages supplied in the command line arguments

package main

import (
	"fmt"
	"net/http"
	"os"
	"strconv"
	"strings"

	"golang.org/x/net/html"
)

type language string

type corpus struct {
	lang1, lang2 string

	name string

	documents    int
	sentences    int
	lang1_tokens int
	lang2_tokens int

	sentence_pairs_download_link string
	lang1_tokens_download_link   string
	lang2_tokens_download_link   string
}

func printHTMLNode(node *html.Node, depth int) {
	fmt.Printf("%s %q\n", strings.Repeat("\t", depth), node.Data)

	for child := node.FirstChild; child != nil; child = child.NextSibling {
		printHTMLNode(child, depth+1)
	}
}

func parseNumberWithPowerSuffix(stringRepresentation string) int {
	multiplier := 10
	switch stringRepresentation[len(stringRepresentation)-1] {
	case 'M':
		multiplier = 1000000
	case 'k':
		multiplier = 1000
	}
	adjusted, _ := strconv.ParseFloat(stringRepresentation[:len(stringRepresentation)-1], 32)
	adjusted *= float64(multiplier)
	return int(adjusted)
}

// looks up the OPUS page for the given language pair and returns the corpora
func corporaOfSentencePairs(lang1, lang2 language) (*[]corpus, bool) {
	url := fmt.Sprintf("https://opus.nlpl.eu/?src=%s&trg=%s&minsize=all", lang1, lang2)

	resp, err := http.Get(url)
	if err != nil {
		return nil, false
	}

	defer resp.Body.Close()

	root, err := html.Parse(resp.Body)
	if err != nil {
		return nil, false
	}

	// we navigate to the table and then its rows.
	// this navigation is incredibly fragile to the page structure used by OPUS.
	table := root.FirstChild.NextSibling.FirstChild.NextSibling.NextSibling.FirstChild.NextSibling.NextSibling.NextSibling.NextSibling.NextSibling.NextSibling.NextSibling.NextSibling.NextSibling.NextSibling.NextSibling.NextSibling.NextSibling.NextSibling.NextSibling.NextSibling.NextSibling
	rows := table.FirstChild.FirstChild

	corpora := make([]corpus, 0)

	for row := rows.FirstChild.NextSibling; row != nil; row = row.NextSibling {
		if row.Data != "tr" {
			continue
		}

		corp := new(corpus)
		corp.lang1 = string(lang1)
		corp.lang2 = string(lang2)

		var version string

		cell := row.FirstChild
		if cell.FirstChild != nil && cell.FirstChild.FirstChild != nil && cell.FirstChild.FirstChild.FirstChild != nil {
			corp.name = cell.FirstChild.FirstChild.FirstChild.Data

			// the first character is a line feed, and the second is also unimportant; we ignore them
			version = cell.FirstChild.FirstChild.NextSibling.Data[2:]
		}

		cell = cell.NextSibling
		if cell.FirstChild != nil {
			corp.documents, _ = strconv.Atoi(cell.FirstChild.Data)
		}

		cell = cell.NextSibling
		if cell.FirstChild != nil {
			count := cell.FirstChild.Data
			corp.sentences = parseNumberWithPowerSuffix(count)
		}

		cell = cell.NextSibling
		if cell.FirstChild != nil {
			corp.lang1_tokens = parseNumberWithPowerSuffix(cell.FirstChild.Data)
		}

		cell = cell.NextSibling
		if cell.FirstChild != nil {
			corp.lang2_tokens = parseNumberWithPowerSuffix(cell.FirstChild.Data)
		}

		corp.sentence_pairs_download_link = fmt.Sprintf("https://object.pouta.csc.fi/OPUS-%s/%s/moses/%s-%s.txt.zip", corp.name, version, lang1, lang2)
		corp.lang1_tokens_download_link = fmt.Sprintf("https://object.pouta.csc.fi/OPUS-%s/%s/mono/%s.tok.gz", corp.name, version, lang1)
		corp.lang2_tokens_download_link = fmt.Sprintf("https://object.pouta.csc.fi/OPUS-%s/%s/mono/%s.tok.gz", corp.name, version, lang2)

		corpora = append(corpora, *corp)
	}

	return &corpora, true
}

func main() {
	languages := []language{}
	for _, lang := range os.Args[1:] {
		languages = append(languages, language(lang))
	}

	corpora, ok := corporaOfSentencePairs(languages[0], languages[1])
	if !ok {
		return
	}

	fmt.Printf("%v\n", corpora)
}
