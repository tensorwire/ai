package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/open-ai-org/gguf"
	"github.com/open-ai-org/tokenizer"
)

func cmdInferImpl(model, prompt string) {
	path := resolveModel(model)

	configPath := filepath.Join(path, "config.json")
	configData, err := os.ReadFile(configPath)
	if err != nil {
		log.Fatalf("No config.json in %s", path)
	}
	var cfg map[string]interface{}
	json.Unmarshal(configData, &cfg)

	dim := int(cfg["hidden_size"].(float64))
	layers := int(cfg["num_hidden_layers"].(float64))
	heads := int(cfg["num_attention_heads"].(float64))
	vocabSize := int(cfg["vocab_size"].(float64))
	ffnDim := int(cfg["intermediate_size"].(float64))
	maxSeq := 2048
	if v, ok := cfg["max_position_embeddings"]; ok {
		maxSeq = int(v.(float64))
	}
	numKVHeads := heads
	if v, ok := cfg["num_key_value_heads"]; ok {
		numKVHeads = int(v.(float64))
	}
	ropeTheta := float32(10000.0)
	if v, ok := cfg["rope_theta"]; ok {
		ropeTheta = float32(v.(float64))
	}
	act := "silu"
	if v, ok := cfg["hidden_act"]; ok {
		act = v.(string)
	}
	normEps := float32(1e-6)
	if v, ok := cfg["rms_norm_eps"]; ok {
		normEps = float32(v.(float64))
	}

	headDim := dim / heads
	kvDim := numKVHeads * headDim

	fmt.Printf("Model: %s (%d layers, dim=%d, heads=%d, kv=%d, ffn=%d, vocab=%d, act=%s)\n",
		model, layers, dim, heads, numKVHeads, ffnDim, vocabSize, act)

	st, err := gguf.OpenSafeTensors(path)
	if err != nil {
		log.Fatalf("Can't open model: %v", err)
	}

	isAWQ := false
	awqGroupSize := 128
	if qcfg, ok := cfg["quantization_config"]; ok {
		if qm, ok := qcfg.(map[string]interface{}); ok {
			if method, _ := qm["quant_method"].(string); method == "awq" || method == "gptq" {
				isAWQ = true
				if gs, ok := qm["group_size"].(float64); ok {
					awqGroupSize = int(gs)
				}
				fmt.Printf("Quantization: %s (4-bit, group_size=%d)\n", method, awqGroupSize)
			}
		}
	}

	readWeight := func(name string, expectedRows, expectedCols int) ([]float32, int, int, error) {
		if isAWQ {
			qname := strings.TrimSuffix(name, "weight") + "qweight"
			if st.HasTensor(qname) {
				prefix := strings.TrimSuffix(name, "weight")
				return st.DequantAWQ(prefix, awqGroupSize)
			}
		}
		data, _, err := st.ReadTensorFloat32(name)
		if err != nil {
			return nil, 0, 0, err
		}
		return data, expectedRows, expectedCols, nil
	}

	tok, tokErr := tokenizer.LoadTokenizer(path)
	if tokErr != nil {
		log.Fatalf("Tokenizer: %v", tokErr)
	}

	tokens := tok.Encode(prompt)
	fmt.Printf("Prompt: %q → %d tokens\n", prompt, len(tokens))

	fmt.Print("Loading embeddings... ")
	embedData, _, _ := st.ReadTensorFloat32("model.embed_tokens.weight")
	fmt.Println("done")

	var lmHeadData []float32
	lmHeadData, _, err = st.ReadTensorFloat32("lm_head.weight")
	if err != nil {
		lmHeadData = embedData
	}

	eng := selectEngine("auto")
	fmt.Printf("Engine: %s\n", eng.Name())

	mv := func(out, W, x []float32, rows, cols int) {
		r := eng.MatMul(W, x, rows, cols, 1)
		copy(out, r)
	}

	x := make([]float32, dim)
	buf := make([]float32, dim)
	q := make([]float32, dim)
	k := make([]float32, kvDim)
	v := make([]float32, kvDim)
	att := make([]float32, heads*maxSeq)
	attnOut := make([]float32, dim)
	ffnBuf := make([]float32, ffnDim)
	ffnBuf2 := make([]float32, ffnDim)
	keyCache := make([][]float32, layers)
	valCache := make([][]float32, layers)
	for l := 0; l < layers; l++ {
		keyCache[l] = make([]float32, maxSeq*kvDim)
		valCache[l] = make([]float32, maxSeq*kvDim)
	}

	forward := func(tokenID, pos int) []float32 {
		tokOff := tokenID * dim
		if tokOff+dim > len(embedData) {
			return nil
		}
		copy(x, embedData[tokOff:tokOff+dim])
		for i := range buf { buf[i] = 0 }
		for i := range q { q[i] = 0 }
		for i := range k { k[i] = 0 }
		for i := range v { v[i] = 0 }
		for i := range att { att[i] = 0 }
		for i := range attnOut { attnOut[i] = 0 }
		for i := range ffnBuf { ffnBuf[i] = 0 }
		for i := range ffnBuf2 { ffnBuf2[i] = 0 }

		for l := 0; l < layers; l++ {
			prefix := fmt.Sprintf("model.layers.%d.", l)

			normW, _, _ := st.ReadTensorFloat32(prefix + "input_layernorm.weight")
			copy(buf, x)
			eng.RMSNorm(buf, normW, normEps)

			wq, qRows, qCols, _ := readWeight(prefix+"self_attn.q_proj.weight", dim, dim)
			wk, kRows, kCols, _ := readWeight(prefix+"self_attn.k_proj.weight", kvDim, dim)
			wv, vRows, vCols, _ := readWeight(prefix+"self_attn.v_proj.weight", kvDim, dim)

			mv(q, wq, buf, qRows, qCols)
			mv(k[:kvDim], wk, buf, kRows, kCols)
			mv(v[:kvDim], wv, buf, vRows, vCols)

			if bq, _, e := st.ReadTensorFloat32(prefix + "self_attn.q_proj.bias"); e == nil {
				for i := range bq { q[i] += bq[i] }
			}
			if bk, _, e := st.ReadTensorFloat32(prefix + "self_attn.k_proj.bias"); e == nil {
				for i := range bk { k[i] += bk[i] }
			}
			if bv, _, e := st.ReadTensorFloat32(prefix + "self_attn.v_proj.bias"); e == nil {
				for i := range bv { v[i] += bv[i] }
			}

			applyRoPE(q, k[:kvDim], pos, headDim, ropeTheta, heads, numKVHeads)

			copy(keyCache[l][pos*kvDim:(pos+1)*kvDim], k[:kvDim])
			copy(valCache[l][pos*kvDim:(pos+1)*kvDim], v[:kvDim])

			kvMul := heads / numKVHeads
			for i := range attnOut[:dim] { attnOut[i] = 0 }
			for h := 0; h < heads; h++ {
				qOff := h * headDim
				kvH := h / kvMul
				kvOff := kvH * headDim
				scale := float32(1.0 / math.Sqrt(float64(headDim)))
				for t := 0; t <= pos; t++ {
					var dot float32
					for j := 0; j < headDim; j++ {
						dot += q[qOff+j] * keyCache[l][t*kvDim+kvOff+j]
					}
					att[h*(pos+1)+t] = dot * scale
				}
				softmax(att[h*(pos+1):h*(pos+1)+(pos+1)], pos+1)
				for t := 0; t <= pos; t++ {
					w := att[h*(pos+1)+t]
					for j := 0; j < headDim; j++ {
						attnOut[qOff+j] += w * valCache[l][t*kvDim+kvOff+j]
					}
				}
			}

			wo, woRows, woCols, _ := readWeight(prefix+"self_attn.o_proj.weight", dim, dim)
			proj := make([]float32, dim)
			mv(proj, wo, attnOut, woRows, woCols)
			for i := 0; i < dim; i++ { x[i] += proj[i] }

			normW2, _, _ := st.ReadTensorFloat32(prefix + "post_attention_layernorm.weight")
			copy(buf, x)
			eng.RMSNorm(buf, normW2, normEps)

			gate, gRows, gCols, _ := readWeight(prefix+"mlp.gate_proj.weight", ffnDim, dim)
			up, uRows, uCols, _ := readWeight(prefix+"mlp.up_proj.weight", ffnDim, dim)
			down, dRows, dCols, _ := readWeight(prefix+"mlp.down_proj.weight", dim, ffnDim)
			mv(ffnBuf, gate, buf, gRows, gCols)
			mv(ffnBuf2, up, buf, uRows, uCols)
			if act == "relu" {
				for i := 0; i < ffnDim; i++ {
					if ffnBuf[i] < 0 { ffnBuf[i] = 0 }
					ffnBuf[i] *= ffnBuf2[i]
				}
			} else {
				for i := 0; i < ffnDim; i++ {
					ffnBuf[i] = silu(ffnBuf[i]) * ffnBuf2[i]
				}
			}
			downOut := make([]float32, dim)
			mv(downOut, down, ffnBuf, dRows, dCols)
			for i := 0; i < dim; i++ { x[i] += downOut[i] }
		}

		finalNorm, _, _ := st.ReadTensorFloat32("model.norm.weight")
		eng.RMSNorm(x, finalNorm, normEps)

		logits := make([]float32, vocabSize)
		mv(logits, lmHeadData, x, vocabSize, dim)
		return logits
	}

	temperature := float32(0.7)
	topK := 40

	fmt.Print("Prefilling... ")
	var logits []float32
	for i, tid := range tokens {
		logits = forward(tid, i)
	}
	fmt.Println("done")

	nextToken := sampleTopK(logits, temperature, topK)

	maxTokens := 200
	fmt.Printf("\nGenerating (max %d tokens):\n", maxTokens)
	fmt.Print(prompt)

	allTokens := make([]int, len(tokens))
	copy(allTokens, tokens)
	allTokens = append(allTokens, nextToken)
	fmt.Print(tok.Decode([]int{nextToken}))
	t0 := time.Now()

	for step := 1; step < maxTokens; step++ {
		pos := len(allTokens) - 1
		lastToken := allTokens[pos]

		logits = forward(lastToken, pos)
		if logits == nil { break }

		nextToken := sampleTopK(logits, temperature, topK)
		allTokens = append(allTokens, nextToken)

		word := tok.Decode([]int{nextToken})
		fmt.Print(word)

		if nextToken == tok.EOS || nextToken == 0 { break }
	}

	elapsed := time.Since(t0)
	genTokens := len(allTokens) - len(tokens)
	fmt.Printf("\n\n--- %d tokens in %v (%.1f tok/s) ---\n", genTokens, elapsed,
		float64(genTokens)/elapsed.Seconds())
}
