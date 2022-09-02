// downloads all the pairs for the languages supplied in the command line arguments

package main

import (
	"bytes"
	"compress/gzip"
	"encoding/xml"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"sort"
	"strconv"

	"golang.org/x/net/html"
)

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

type TMX struct {
	Body TMXBody `xml:"body"`
}

type TMXBody struct {
	Pairs []Pair `xml:"tu"`
}

type Pair struct {
	Sentences []Sentence `xml:"tuv"`
}

type Sentence struct {
	Lang    string `xml:"lang,attr"`
	Content string `xml:"seg"`
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
func corporaOfSentencePairs(lang1, lang2 string) (*[]corpus, bool) {
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

	// the last row is a margin row which shows the totals for each column.
	// we don't want to look at that last now, so we use a row.NextSibling != nil check to avoid it.
	for row := rows.FirstChild.NextSibling; row.NextSibling != nil; row = row.NextSibling {
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

		corp.sentence_pairs_download_link = fmt.Sprintf("https://object.pouta.csc.fi/OPUS-%s/%s/tmx/%s-%s.tmx.gz", corp.name, version, lang1, lang2)
		corp.lang1_tokens_download_link = fmt.Sprintf("https://object.pouta.csc.fi/OPUS-%s/%s/mono/%s.tok.gz", corp.name, version, lang1)
		corp.lang2_tokens_download_link = fmt.Sprintf("https://object.pouta.csc.fi/OPUS-%s/%s/mono/%s.tok.gz", corp.name, version, lang2)

		corpora = append(corpora, *corp)
	}

	return &corpora, true
}

func downloadCorpora(corpora *[]corpus) []Pair {
	pairs := make(chan []byte, len(*corpora))

	for _, corp := range *corpora {
		func(corp corpus, pairs chan<- []byte) {
			resp, err := http.Get(corp.sentence_pairs_download_link)
			if err != nil {
				pairs <- nil
				return
			}
			defer resp.Body.Close()

			body, err := ioutil.ReadAll(resp.Body)
			if err != nil {
				pairs <- nil
				return
			}

			reader := bytes.NewReader(body)
			gzreader, e1 := gzip.NewReader(reader)
			if e1 != nil {
				pairs <- nil
				return
			}

			output, e2 := ioutil.ReadAll(gzreader)
			if e2 != nil {
				pairs <- nil
				return
			}

			pairs <- output
		}(corp, pairs)
	}

	concatenatedPairs := make([]Pair, 0)
	for i := 0; i < len(*corpora); i++ {
		var tmx TMX
		p := <-pairs
		if p != nil {
			xml.Unmarshal(p, &tmx)
			concatenatedPairs = append(concatenatedPairs, tmx.Body.Pairs...)
		}
	}
	return concatenatedPairs
}

func downloadSentencesPairToFile(lang1, lang2 string, done chan<- struct{}) {
	corpora, ok := corporaOfSentencePairs(lang1, lang2)
	if !ok {
		return
	}

	downloadCorpora(corpora)
	done <- struct{}{}
}

func main() {
	languages := os.Args[1:]
	expected := len(languages) * (len(languages) - 1) / 2

	sort.Strings(languages)

	done := make(chan struct{}, expected)
	for i, lang1 := range languages {
		for _, lang2 := range languages[i+1:] {
			downloadSentencesPairToFile(lang1, lang2, done)
		}
	}

	for i := 0; i < expected; i++ {
		<-done
	}
}
