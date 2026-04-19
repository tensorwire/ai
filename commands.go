package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/open-ai-org/gguf"
	"github.com/open-ai-org/mongoose"
)

var modelsDir string

func init() {
	home, _ := os.UserHomeDir()
	modelsDir = filepath.Join(home, ".tesseract", "models")
	os.MkdirAll(modelsDir, 0755)
}


// === convert ===

func cmdConvert(format string, args []string) {
	if format != "gguf" {
		log.Fatalf("Unknown format: %s (supported: gguf)", format)
	}
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "Usage: mongoose convert gguf <model-dir> [output.gguf] [--quant f32|q8_0|q4_0]")
		os.Exit(1)
	}

	modelDir := resolveModel(args[0])
	outputPath := filepath.Join(modelDir, "model.gguf")
	quantType := "f32"

	// Parse remaining args
	for i := 1; i < len(args); i++ {
		if args[i] == "--quant" && i+1 < len(args) {
			quantType = args[i+1]
			i++
		} else if !strings.HasPrefix(args[i], "--") {
			outputPath = args[i]
		}
	}

	fmt.Printf("Converting %s → %s (quant: %s)\n", modelDir, outputPath, quantType)
	if err := gguf.ConvertSafetensorsToGGUF(modelDir, outputPath, quantType); err != nil {
		log.Fatalf("Convert failed: %v", err)
	}

	fi, _ := os.Stat(outputPath)
	fmt.Printf("Done. %s (%.1f MB)\n", outputPath, float64(fi.Size())/1024/1024)
	fmt.Println("Load in Ollama: ollama create mymodel -f Modelfile")
}

// === pull ===

func cmdPull(model string) {
	// model can be "org/name" (HuggingFace) or just "name" (search local)
	if !strings.Contains(model, "/") {
		fmt.Fprintf(os.Stderr, "Please specify full HuggingFace model: org/name\n")
		fmt.Fprintf(os.Stderr, "  e.g. mongoose pull SparseLLM/ReluLLaMA-7B\n")
		os.Exit(1)
	}

	destDir := filepath.Join(modelsDir, filepath.Base(model))
	os.MkdirAll(destDir, 0755)

	fmt.Printf("Pulling %s → %s\n", model, destDir)

	// First, get the file list from HuggingFace API
	apiURL := fmt.Sprintf("https://huggingface.co/api/models/%s", model)
	resp, err := http.Get(apiURL)
	if err != nil {
		log.Fatalf("Failed to query HuggingFace: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		log.Fatalf("Model not found: %s (HTTP %d)", model, resp.StatusCode)
	}

	var modelInfo struct {
		Siblings []struct {
			Filename string `json:"rfilename"`
		} `json:"siblings"`
	}
	json.NewDecoder(resp.Body).Decode(&modelInfo)

	// Download safetensors files, config.json, tokenizer files
	wanted := []string{}
	for _, f := range modelInfo.Siblings {
		name := f.Filename
		if strings.HasSuffix(name, ".safetensors") ||
			name == "config.json" ||
			name == "tokenizer.json" ||
			name == "tokenizer_config.json" ||
			name == "tokenizer.model" ||
			name == "special_tokens_map.json" ||
			strings.HasSuffix(name, ".safetensors.index.json") {
			wanted = append(wanted, name)
		}
	}

	if len(wanted) == 0 {
		log.Fatalf("No safetensors files found in %s", model)
	}

	fmt.Printf("Files to download: %d\n", len(wanted))
	for _, name := range wanted {
		destPath := filepath.Join(destDir, name)
		if _, err := os.Stat(destPath); err == nil {
			fmt.Printf("  %s (exists, skipping)\n", name)
			continue
		}

		url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", model, name)
		fmt.Printf("  %s ... ", name)

		err := downloadFile(url, destPath)
		if err != nil {
			fmt.Printf("FAILED: %v\n", err)
		} else {
			fi, _ := os.Stat(destPath)
			fmt.Printf("%.1f MB\n", float64(fi.Size())/1024/1024)
		}
	}

	fmt.Printf("\nDone. Use: mongoose info %s\n", filepath.Base(model))
}

func downloadFile(url, dest string) error {
	os.MkdirAll(filepath.Dir(dest), 0755)
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	f, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = io.Copy(f, resp.Body)
	return err
}

// === models ===

func cmdModels() {
	entries, err := os.ReadDir(modelsDir)
	if err != nil || len(entries) == 0 {
		fmt.Println("No models downloaded yet.")
		fmt.Println("  mongoose pull SparseLLM/ReluLLaMA-7B")
		return
	}

	fmt.Println("Downloaded models:")
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		path := filepath.Join(modelsDir, e.Name())
		st, err := gguf.OpenSafeTensors(path)
		if err != nil {
			fmt.Printf("  %s (error: %v)\n", e.Name(), err)
			continue
		}

		// Try to read config.json for architecture info
		arch := "unknown"
		configPath := filepath.Join(path, "config.json")
		if data, err := os.ReadFile(configPath); err == nil {
			var cfg map[string]interface{}
			json.Unmarshal(data, &cfg)
			if a, ok := cfg["architectures"]; ok {
				if arr, ok := a.([]interface{}); ok && len(arr) > 0 {
					arch = fmt.Sprintf("%v", arr[0])
				}
			}
		}

		fmt.Printf("  %-30s %d tensors  %s\n", e.Name(), len(st.TensorNames), arch)
	}
}

// === info ===

func cmdInfo(model string) {
	path := resolveModel(model)

	st, err := gguf.OpenSafeTensors(path)
	if err != nil {
		log.Fatalf("Can't open model: %v", err)
	}

	// Read config.json
	configPath := filepath.Join(path, "config.json")
	configData, _ := os.ReadFile(configPath)
	var cfg map[string]interface{}
	json.Unmarshal(configData, &cfg)

	fmt.Printf("Model: %s\n", model)
	fmt.Printf("Path:  %s\n", path)
	fmt.Printf("Tensors: %d\n", len(st.TensorNames))

	if cfg != nil {
		if v, ok := cfg["hidden_size"]; ok {
			fmt.Printf("Hidden dim: %.0f\n", v)
		}
		if v, ok := cfg["num_hidden_layers"]; ok {
			fmt.Printf("Layers: %.0f\n", v)
		}
		if v, ok := cfg["num_attention_heads"]; ok {
			fmt.Printf("Heads: %.0f\n", v)
		}
		if v, ok := cfg["intermediate_size"]; ok {
			fmt.Printf("FFN dim: %.0f\n", v)
		}
		if v, ok := cfg["vocab_size"]; ok {
			fmt.Printf("Vocab: %.0f\n", v)
		}
		if v, ok := cfg["hidden_act"]; ok {
			fmt.Printf("Activation: %v\n", v)
		}
	}

	// Show first layer's tensors
	fmt.Println("\nLayer 0 weights:")
	for _, name := range st.ListTensors("model.layers.0.") {
		ti, _ := st.GetInfo(name)
		short := strings.TrimPrefix(name, "model.layers.0.")
		fmt.Printf("  %-40s %s %v\n", short, ti.Dtype, ti.Shape)
	}
}

// === bench ===

func cmdBench() {
	fmt.Println("Tesseract GPU Benchmark")
	fmt.Println()

	eng := selectEngine("auto")
	fmt.Printf("Engine: %s\n", eng.Name())

	for _, dim := range []int{512, 1024, 2048} {
		a := make([]float32, dim*dim)
		b := make([]float32, dim*dim)
		for i := range a {
			a[i] = 0.001 * float32(i%1000)
			b[i] = 0.001 * float32(i%997)
		}

		eng.MatMul(a, b, dim, dim, dim)

		start := time.Now()
		iters := 20
		for i := 0; i < iters; i++ {
			eng.MatMul(a, b, dim, dim, dim)
		}
		elapsed := time.Since(start)
		gflops := float64(2*dim*dim*dim*iters) / elapsed.Seconds() / 1e9
		fmt.Printf("  %dx%d: %.1f GFLOPS (%.1f ms/iter)\n",
			dim, dim, gflops, float64(elapsed.Milliseconds())/float64(iters))
	}
}

// === gpus ===

func cmdGPUs() {
	fmt.Println("Detected compute backends:")
	fmt.Println()

	cuda := mongoose.NewCUDA()
	if cuda != nil {
		fmt.Printf("  CUDA: %s\n", cuda.Name())
	}
	cpu := &mongoose.CPU{}
	fmt.Printf("  CPU:  %s (%.1f GFLOPS)\n", cpu.Name(), cpu.Benchmark())

	if cuda != nil {
		fmt.Println("\nScheduler calibration:")
		sched := mongoose.NewScheduler(cuda, cpu)
		sched.CalibrateMatMul(4096, 4096, 1)
		sched.CalibrateAll(mongoose.NormKey(4096), func(eng mongoose.Engine) {
			x := make([]float32, 4096)
			w := make([]float32, 4096)
			for i := range w {
				w[i] = 1.0
			}
			eng.RMSNorm(x, w, 1e-6)
		})
		fmt.Print(sched.String())
	}
}

// === infer ===

func cmdInfer(model string, args []string) {
	prompt := strings.Join(args, " ")
	cmdInferImpl(model, prompt)
}


// === export ===

func cmdExport(args []string) {
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "Usage: mongoose export qdrant <url> <collection> [output.jsonl]")
		fmt.Fprintln(os.Stderr, "       mongoose export npy <input.npy> <output.jsonl>")
		os.Exit(1)
	}

	switch args[0] {
	case "qdrant":
		if len(args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: mongoose export qdrant <url> <collection> [output.jsonl]")
			os.Exit(1)
		}
		qdrantURL := args[1]
		collection := args[2]
		output := collection + ".jsonl"
		if len(args) >= 4 {
			output = args[3]
		}
		exportQdrant(qdrantURL, collection, output)

	case "npy":
		if len(args) < 2 {
			fmt.Fprintln(os.Stderr, "Usage: mongoose export npy <input.npy> [output.jsonl]")
			os.Exit(1)
		}
		input := args[1]
		output := strings.TrimSuffix(input, ".npy") + ".jsonl"
		if len(args) >= 3 {
			output = args[2]
		}
		exportNpy(input, output)

	default:
		fmt.Fprintf(os.Stderr, "Unknown export source: %s (supported: qdrant, npy)\n", args[0])
		os.Exit(1)
	}
}

func exportQdrant(baseURL, collection, output string) {
	fmt.Printf("Exporting %s/%s → %s\n", baseURL, collection, output)

	f, err := os.Create(output)
	if err != nil {
		log.Fatalf("Create output: %v", err)
	}
	defer f.Close()

	total := 0
	var offsetStr string // empty for first page

	for {
		// Build scroll request
		body := `{"limit": 100, "with_payload": true, "with_vector": false`
		if offsetStr != "" {
			body += `, "offset": ` + offsetStr
		}
		body += "}"

		url := fmt.Sprintf("%s/collections/%s/points/scroll", baseURL, collection)
		resp, err := http.Post(url, "application/json", strings.NewReader(body))
		if err != nil {
			log.Fatalf("Qdrant request: %v", err)
		}

		var result struct {
			Result struct {
				Points []struct {
					ID      interface{}            `json:"id"`
					Payload map[string]interface{} `json:"payload"`
				} `json:"points"`
				NextPageOffset interface{} `json:"next_page_offset"`
			} `json:"result"`
		}
		json.NewDecoder(resp.Body).Decode(&result)
		resp.Body.Close()

		for _, p := range result.Result.Points {
			line, _ := json.Marshal(p.Payload)
			f.Write(line)
			f.Write([]byte("\n"))
			total++
		}

		if result.Result.NextPageOffset == nil || len(result.Result.Points) == 0 {
			break
		}
		// Format offset as raw JSON number (large int)
		raw, _ := json.Marshal(result.Result.NextPageOffset)
		offsetStr = string(raw)
		fmt.Printf("\r  %d points...", total)
	}

	fmt.Printf("\r  %d points exported to %s\n", total, output)
}

func exportNpy(input, output string) {
	fmt.Printf("Exporting %s → %s\n", input, output)

	arr, err := gguf.ReadNpy(input)
	if err != nil {
		log.Fatalf("Read npy: %v", err)
	}
	fmt.Printf("  Shape: %v, Dtype: %s\n", arr.Shape, arr.Dtype)

	f, err := os.Create(output)
	if err != nil {
		log.Fatalf("Create output: %v", err)
	}
	defer f.Close()

	for i := 0; i < arr.Rows(); i++ {
		row := arr.Row(i)
		line, _ := json.Marshal(row)
		f.Write(line)
		f.Write([]byte("\n"))
	}

	fmt.Printf("  %d rows exported\n", arr.Rows())
}

// === helpers ===

func resolveModel(name string) string {
	// Try as absolute path first
	if _, err := os.Stat(name); err == nil {
		return name
	}
	// Try in models dir
	path := filepath.Join(modelsDir, name)
	if _, err := os.Stat(path); err == nil {
		return path
	}
	// Try with common suffixes
	for _, suffix := range []string{"", "-hf", "-chat"} {
		p := filepath.Join(modelsDir, name+suffix)
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	log.Fatalf("Model not found: %s\nTry: mongoose pull <org>/%s", name, name)
	return ""
}


func softmax(x []float32, n int) {
	max := x[0]
	for i := 1; i < n; i++ {
		if x[i] > max {
			max = x[i]
		}
	}
	var sum float32
	for i := 0; i < n; i++ {
		x[i] = float32(math.Exp(float64(x[i] - max)))
		sum += x[i]
	}
	inv := 1.0 / sum
	for i := 0; i < n; i++ {
		x[i] *= inv
	}
}

func silu(x float32) float32 {
	return x * float32(1.0/(1.0+math.Exp(-float64(x))))
}

func applyRoPE(q, k []float32, pos, headDim int, theta float32, numHeads, numKVHeads int) {
	// HuggingFace rotate_half convention: pair (x[i], x[i+halfDim]) not (x[2i], x[2i+1])
	half := headDim / 2
	for h := 0; h < numHeads; h++ {
		base := h * headDim
		for i := 0; i < half; i++ {
			freq := float32(1.0 / math.Pow(float64(theta), float64(2*i)/float64(headDim)))
			val := float32(pos) * freq
			cos := float32(math.Cos(float64(val)))
			sin := float32(math.Sin(float64(val)))
			q0 := q[base+i]
			q1 := q[base+half+i]
			q[base+i] = q0*cos - q1*sin
			q[base+half+i] = q0*sin + q1*cos
		}
	}
	for h := 0; h < numKVHeads; h++ {
		base := h * headDim
		for i := 0; i < half; i++ {
			freq := float32(1.0 / math.Pow(float64(theta), float64(2*i)/float64(headDim)))
			val := float32(pos) * freq
			cos := float32(math.Cos(float64(val)))
			sin := float32(math.Sin(float64(val)))
			k0 := k[base+i]
			k1 := k[base+half+i]
			k[base+i] = k0*cos - k1*sin
			k[base+half+i] = k0*sin + k1*cos
		}
	}
}

func argmax(x []float32) int {
	best := 0
	for i := 1; i < len(x); i++ {
		if x[i] > x[best] {
			best = i
		}
	}
	return best
}

// sampleTopK picks a token using temperature-scaled top-k sampling.
// temp=0 → greedy argmax. topK=0 → sample from full distribution.
func sampleTopK(logits []float32, temp float32, topK int) int {
	if temp <= 0 {
		return argmax(logits)
	}

	// Apply temperature
	inv := 1.0 / temp
	for i := range logits {
		logits[i] *= inv
	}

	// Find top-k indices
	n := len(logits)
	if topK <= 0 || topK > n {
		topK = n
	}

	type kv struct {
		idx int
		val float32
	}
	topItems := make([]kv, topK)
	for i := 0; i < topK; i++ {
		topItems[i] = kv{-1, -math.MaxFloat32}
	}
	for i, v := range logits {
		if v > topItems[topK-1].val {
			topItems[topK-1] = kv{i, v}
			// Insertion sort to keep sorted
			for j := topK - 1; j > 0 && topItems[j].val > topItems[j-1].val; j-- {
				topItems[j], topItems[j-1] = topItems[j-1], topItems[j]
			}
		}
	}

	// Softmax over top-k
	maxVal := topItems[0].val
	var sum float32
	probs := make([]float32, topK)
	for i, item := range topItems {
		if item.idx < 0 {
			topK = i
			break
		}
		probs[i] = float32(math.Exp(float64(item.val - maxVal)))
		sum += probs[i]
	}
	for i := range probs[:topK] {
		probs[i] /= sum
	}

	// Sample
	r := rand.Float32()
	var cum float32
	for i := 0; i < topK; i++ {
		cum += probs[i]
		if r < cum {
			return topItems[i].idx
		}
	}
	return topItems[0].idx
}

