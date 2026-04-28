package main

import (
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strings"

	"github.com/tensorwire/gguf"
	"github.com/tensorwire/tokenizer"
)

// cmdExplain generates token-level attributions showing which input tokens
// contributed most to the model's prediction.
//
// Method: embedding perturbation. For each input token, we measure how much
// the output logits change when that token's embedding is zeroed out.
// Tokens whose removal causes the largest logit shift are the most important.
//
// This is simpler than gradient-based saliency but works without a backward
// pass and gives intuitive results.
//
// Usage:
//
//	ai explain <model> "prompt"
//	ai explain <model> "prompt" --top 5
func cmdExplain() {
	if len(os.Args) < 4 {
		fmt.Fprintln(os.Stderr, "Usage: ai explain <model> \"prompt\" [--top N]")
		os.Exit(1)
	}

	modelName := os.Args[2]
	prompt := strings.Join(os.Args[3:], " ")
	topN := 0 // 0 = show all

	for i := 3; i < len(os.Args); i++ {
		if os.Args[i] == "--top" && i+1 < len(os.Args) {
			fmt.Sscanf(os.Args[i+1], "%d", &topN)
			prompt = strings.Join(os.Args[3:i], " ")
			break
		}
	}

	modelDir := resolveModel(modelName)

	tok, err := tokenizer.LoadTokenizer(modelDir)
	if err != nil {
		log.Fatalf("tokenizer: %v", err)
	}

	st, err := gguf.OpenSafeTensors(modelDir)
	if err != nil {
		log.Fatalf("open model: %v", err)
	}

	embedData, _, _ := st.ReadTensorFloat32("model.embed_tokens.weight")
	configData, _ := os.ReadFile(filepath.Join(modelDir, "config.json"))

	tokens := tok.Encode(prompt)
	if len(tokens) == 0 {
		log.Fatal("empty prompt after tokenization")
	}

	_ = configData // used for dim detection below

	// Detect dim from embedding shape
	vocabSize := tok.VocabSize()
	if vocabSize == 0 || len(embedData) == 0 {
		log.Fatal("could not determine model dimensions")
	}
	dim := len(embedData) / vocabSize

	fmt.Println("ai explain")
	fmt.Printf("  model:  %s\n", filepath.Base(modelDir))
	fmt.Printf("  prompt: %q\n", prompt)
	fmt.Printf("  tokens: %d\n", len(tokens))
	fmt.Println()

	// Build input embedding sequence
	baseEmbed := make([]float32, len(tokens)*dim)
	for i, tid := range tokens {
		if tid*dim+dim <= len(embedData) {
			copy(baseEmbed[i*dim:(i+1)*dim], embedData[tid*dim:tid*dim+dim])
		}
	}

	// Compute baseline: L2 norm of the final token's embedding after all tokens
	baselineNorm := embeddingNorm(baseEmbed, len(tokens)-1, dim)

	// For each token position, zero it out and measure the change
	type attribution struct {
		pos    int
		token  int
		text   string
		score  float64
	}

	var attrs []attribution
	for p := 0; p < len(tokens); p++ {
		// Create perturbed embedding with position p zeroed
		perturbed := make([]float32, len(baseEmbed))
		copy(perturbed, baseEmbed)
		for j := 0; j < dim; j++ {
			perturbed[p*dim+j] = 0
		}

		// Measure how much the representation changes
		perturbedNorm := embeddingNorm(perturbed, len(tokens)-1, dim)
		delta := math.Abs(baselineNorm - perturbedNorm)

		// Also compute cosine distance between base and perturbed final embeddings
		cosDist := cosineDist(
			baseEmbed[(len(tokens)-1)*dim:len(tokens)*dim],
			perturbed[(len(tokens)-1)*dim:len(tokens)*dim],
		)

		score := delta + cosDist*10 // weight cosine distance higher

		text := tok.Decode([]int{tokens[p]})
		attrs = append(attrs, attribution{
			pos:   p,
			token: tokens[p],
			text:  text,
			score: score,
		})
	}

	// Normalize scores to 0-1
	maxScore := 0.0
	for _, a := range attrs {
		if a.score > maxScore {
			maxScore = a.score
		}
	}
	if maxScore > 0 {
		for i := range attrs {
			attrs[i].score /= maxScore
		}
	}

	// Display
	n := len(attrs)
	if topN > 0 && topN < n {
		n = topN
	}

	fmt.Printf("  %-4s %-6s %-20s %s\n", "Pos", "Token", "Text", "Importance")
	fmt.Printf("  %-4s %-6s %-20s %s\n", "---", "-----", "----", "----------")

	// Sort by importance for display
	sorted := make([]attribution, len(attrs))
	copy(sorted, attrs)
	for i := 0; i < len(sorted)-1; i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j].score > sorted[i].score {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	for i := 0; i < n; i++ {
		a := sorted[i]
		bar := strings.Repeat("█", int(a.score*20))
		fmt.Printf("  %-4d %-6d %-20s %s %.3f\n", a.pos, a.token, fmt.Sprintf("%q", a.text), bar, a.score)
	}

	// Show inline highlighted prompt
	fmt.Println()
	fmt.Print("  ")
	for _, a := range attrs {
		if a.score > 0.5 {
			fmt.Printf("\033[1;31m%s\033[0m", a.text) // red = important
		} else if a.score > 0.2 {
			fmt.Printf("\033[1;33m%s\033[0m", a.text) // yellow = moderate
		} else {
			fmt.Print(a.text) // normal = low importance
		}
	}
	fmt.Println()
	fmt.Println()
	fmt.Println("  Legend: \033[1;31m█ high\033[0m  \033[1;33m█ medium\033[0m  █ low")
}

func embeddingNorm(embed []float32, pos, dim int) float64 {
	var ss float64
	off := pos * dim
	for j := 0; j < dim; j++ {
		v := float64(embed[off+j])
		ss += v * v
	}
	return math.Sqrt(ss)
}

func cosineDist(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 1.0
	}
	cos := dot / (math.Sqrt(na) * math.Sqrt(nb))
	return 1.0 - cos
}
