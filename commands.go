package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/tensorwire/gguf"
	"github.com/tensorwire/mongoose"
)

var modelsDir string

func init() {
	home, _ := os.UserHomeDir()
	modelsDir = filepath.Join(home, ".ai", "models")
	os.MkdirAll(modelsDir, 0755)
}


// === convert ===

func cmdConvert(format string, args []string) {
	if format != "gguf" && format != "safetensors" {
		log.Fatalf("Unknown format: %s (supported: gguf, safetensors)", format)
	}
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "Usage: ai convert gguf <model-dir> [output.gguf] [--quant f32|q8_0|q4_0]")
		fmt.Fprintln(os.Stderr, "       ai convert safetensors <model.gguf> [output-dir]")
		os.Exit(1)
	}

	if format == "safetensors" {
		cmdConvertToSafeTensors(args)
		return
	}

	modelDir := resolveModel(args[0])
	outputPath := filepath.Join(modelDir, "model.gguf")
	quantType := "f32"

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

func cmdConvertToSafeTensors(args []string) {
	inputPath := args[0]
	outputDir := ""
	if len(args) >= 2 {
		outputDir = args[1]
	}

	ms, err := OpenModel(inputPath)
	if err != nil {
		log.Fatalf("open model: %v", err)
	}

	if outputDir == "" {
		base := strings.TrimSuffix(filepath.Base(inputPath), filepath.Ext(inputPath))
		outputDir = base + "-safetensors"
	}
	os.MkdirAll(outputDir, 0755)

	names := ms.TensorNames()
	if len(names) == 0 {
		log.Fatal("no tensors found in model")
	}

	fmt.Printf("Converting %s → %s (%d tensors)\n", inputPath, outputDir, len(names))

	// Collect metadata for streaming writer
	type tensorEntry struct {
		name  string
		elems int
		shape []int
	}
	var entries []tensorEntry
	var metas []gguf.TensorMeta

	for _, name := range names {
		data, info, err := ms.ReadTensorFloat32Full(name)
		if err != nil || data == nil {
			log.Printf("WARN: skip %s: %v", name, err)
			continue
		}
		shape := info.Shape
		if len(shape) == 0 {
			shape = []int{len(data)}
		}
		entries = append(entries, tensorEntry{name: name, elems: len(data), shape: shape})
		metas = append(metas, gguf.TensorMeta{Name: name, Shape: shape, Elems: len(data)})
	}

	outPath := filepath.Join(outputDir, "model.safetensors")
	w, err := gguf.NewStreamingSafeTensorsWriter(outPath, metas)
	if err != nil {
		log.Fatalf("create output: %v", err)
	}

	for i, e := range entries {
		data, _ := ms.ReadTensorFloat32(e.name)
		if err := w.WriteTensor(data); err != nil {
			log.Fatalf("write %s: %v", e.name, err)
		}
		if (i+1)%50 == 0 || i == len(entries)-1 {
			fmt.Printf("\r  %d/%d tensors written", i+1, len(entries))
		}
	}
	fmt.Println()
	w.Close()

	// Copy config.json if available alongside the source
	srcDir := ms.Dir()
	for _, f := range []string{"config.json", "tokenizer.json", "tokenizer_config.json",
		"tokenizer.model", "special_tokens_map.json"} {
		src := filepath.Join(srcDir, f)
		if data, err := os.ReadFile(src); err == nil {
			os.WriteFile(filepath.Join(outputDir, f), data, 0644)
		}
	}

	fi, _ := os.Stat(outPath)
	if fi != nil {
		fmt.Printf("Done. %.1f GB → %s\n", float64(fi.Size())/(1024*1024*1024), outputDir)
	}
}

// === pull ===

func cmdPull(model string) {
	// model can be "org/name" (HuggingFace) or just "name" (search local)
	if !strings.Contains(model, "/") {
		fmt.Fprintf(os.Stderr, "Please specify full HuggingFace model: org/name\n")
		fmt.Fprintf(os.Stderr, "  e.g. ai pull SparseLLM/ReluLLaMA-7B\n")
		os.Exit(1)
	}

	destDir := filepath.Join(modelsDir, filepath.Base(model))
	os.MkdirAll(destDir, 0755)

	fmt.Printf("Pulling %s → %s\n", model, destDir)

	// First, get the file list from HuggingFace API
	apiURL := fmt.Sprintf("https://huggingface.co/api/models/%s", model)
	resp, err := hfGet(apiURL)
	if err != nil {
		log.Fatalf("Failed to query HuggingFace: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == 401 || resp.StatusCode == 403 {
		resp.Body.Close()
		if hfToken() != "" {
			fmt.Printf("Access denied for %s — your HuggingFace token doesn't have access to this gated model.\n", model)
			fmt.Println("Accept the license at: https://huggingface.co/" + model)
			fmt.Print("Enter a new token (or press Enter to abort): ")
		} else {
			fmt.Printf("Access denied for %s — this is a gated model requiring a HuggingFace token.\n", model)
			fmt.Println("1. Create a token at: https://huggingface.co/settings/tokens")
			fmt.Println("2. Accept the model license at: https://huggingface.co/" + model)
			fmt.Print("Paste your HuggingFace token (hf_...): ")
		}
		scanner := bufio.NewScanner(os.Stdin)
		if !scanner.Scan() || strings.TrimSpace(scanner.Text()) == "" {
			log.Fatal("Aborted.")
		}
		token := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(token, "hf_") {
			log.Fatal("Invalid token — must start with hf_")
		}
		home, _ := os.UserHomeDir()
		tokenDir := filepath.Join(home, ".cache", "huggingface")
		os.MkdirAll(tokenDir, 0700)
		os.WriteFile(filepath.Join(tokenDir, "token"), []byte(token+"\n"), 0600)
		fmt.Println("Token saved. Retrying...")

		resp, err = hfGet(apiURL)
		if err != nil {
			log.Fatalf("Failed to query HuggingFace: %v", err)
		}
		defer resp.Body.Close()
		if resp.StatusCode == 401 || resp.StatusCode == 403 {
			log.Fatalf("Still denied — make sure you've accepted the license at: https://huggingface.co/%s", model)
		}
		if resp.StatusCode != 200 {
			log.Fatalf("Model not found: %s (HTTP %d)", model, resp.StatusCode)
		}
	}
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

	hasSafeTensors := false
	for _, name := range wanted {
		if strings.HasSuffix(name, ".safetensors") {
			hasSafeTensors = true
			break
		}
	}
	if !hasSafeTensors {
		for _, f := range modelInfo.Siblings {
			name := f.Filename
			if strings.HasSuffix(name, ".gguf") {
				wanted = append(wanted, name)
			}
		}
	}
	if len(wanted) == 0 {
		log.Fatalf("No model files found in %s (no .safetensors or .gguf)", model)
	}
	if !hasSafeTensors {
		fmt.Println("  (no SafeTensors files — downloading GGUF)")
	}

	fmt.Printf("Files to download: %d\n", len(wanted))
	authRetried := false
	for i, name := range wanted {
		destPath := filepath.Join(destDir, name)
		if _, err := os.Stat(destPath); err == nil {
			fmt.Printf("  %s (exists, skipping)\n", name)
			continue
		}

		url := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", model, name)
		fmt.Printf("  %s ... ", name)

		err := downloadFile(url, destPath)
		if err != nil && (strings.Contains(err.Error(), "403") || strings.Contains(err.Error(), "401")) && !authRetried {
			fmt.Println()
			authRetried = true
			licenseURL := "https://huggingface.co/" + model
			tokenURL := "https://huggingface.co/settings/tokens"
			if hfToken() != "" {
				fmt.Printf("\nAccess denied — your token doesn't have access to this gated model.\n")
				fmt.Println("Accept the license at: " + licenseURL)
				if openBrowser(licenseURL) {
					fmt.Println("(opened in browser)")
				}
				fmt.Print("\nEnter a new token (or press Enter to abort): ")
			} else {
				fmt.Printf("\nThis is a gated model requiring a HuggingFace token.\n")
				fmt.Println("1. Create a token at: " + tokenURL)
				fmt.Println("2. Accept the model license at: " + licenseURL)
				if openBrowser(tokenURL) {
					fmt.Println("(opened in browser)")
				}
				fmt.Print("\nPaste your HuggingFace token (hf_...): ")
			}
			scanner := bufio.NewScanner(os.Stdin)
			if !scanner.Scan() || strings.TrimSpace(scanner.Text()) == "" {
				log.Fatal("Aborted.")
			}
			token := strings.TrimSpace(scanner.Text())
			if !strings.HasPrefix(token, "hf_") {
				log.Fatal("Invalid token — must start with hf_")
			}
			home, _ := os.UserHomeDir()
			tokenDir := filepath.Join(home, ".cache", "huggingface")
			os.MkdirAll(tokenDir, 0700)
			os.WriteFile(filepath.Join(tokenDir, "token"), []byte(token+"\n"), 0600)
			fmt.Println("Token saved. Retrying downloads...")

			// Retry all files from the beginning
			for j := 0; j <= i; j++ {
				rName := wanted[j]
				rDest := filepath.Join(destDir, rName)
				if _, err := os.Stat(rDest); err == nil {
					continue
				}
				rURL := fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", model, rName)
				fmt.Printf("  %s ... ", rName)
				if err := downloadFile(rURL, rDest); err != nil {
					fmt.Printf("FAILED: %v\n", err)
				} else {
					fi, _ := os.Stat(rDest)
					fmt.Printf("%.1f MB\n", float64(fi.Size())/1024/1024)
				}
			}
			continue
		} else if err != nil {
			fmt.Printf("FAILED: %v\n", err)
		} else {
			fi, _ := os.Stat(destPath)
			fmt.Printf("%.1f MB\n", float64(fi.Size())/1024/1024)
		}
	}

	fmt.Printf("\nDone. Use: ai info %s\n", filepath.Base(model))
}

func openBrowser(url string) bool {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", url)
	case "linux":
		cmd = exec.Command("xdg-open", url)
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	default:
		return false
	}
	return cmd.Start() == nil
}

func hfToken() string {
	home, _ := os.UserHomeDir()
	for _, p := range []string{
		filepath.Join(home, ".cache", "huggingface", "token"),
		filepath.Join(home, ".huggingface", "token"),
	} {
		if data, err := os.ReadFile(p); err == nil {
			return strings.TrimSpace(string(data))
		}
	}
	return ""
}

func hfGet(url string) (*http.Response, error) {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	if tok := hfToken(); tok != "" {
		req.Header.Set("Authorization", "Bearer "+tok)
	}
	return http.DefaultClient.Do(req)
}

func downloadFile(url, dest string) error {
	os.MkdirAll(filepath.Dir(dest), 0755)
	resp, err := hfGet(url)
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
		fmt.Println("  ai pull SparseLLM/ReluLLaMA-7B")
		return
	}

	fmt.Println("Downloaded models:")
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		path := filepath.Join(modelsDir, e.Name())
		ms, err := OpenModel(path)
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

		fmt.Printf("  %-30s %d tensors  %s\n", e.Name(), len(ms.TensorNames()), arch)
	}
}

// === info ===

func cmdInfo(model string) {
	path := resolveModel(model)

	ms, err := OpenModel(path)
	if err != nil {
		log.Fatalf("Can't open model: %v", err)
	}
	st := ms.ST()

	// Read config.json
	configPath := filepath.Join(path, "config.json")
	configData, _ := os.ReadFile(configPath)
	var cfg map[string]interface{}
	json.Unmarshal(configData, &cfg)

	fmt.Printf("Model: %s\n", model)
	fmt.Printf("Path:  %s\n", path)
	fmt.Printf("Format: %s\n", ms.Format())
	fmt.Printf("Tensors: %d\n", len(ms.TensorNames()))

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

	if st != nil {
		fmt.Println("\nLayer 0 weights:")
		for _, name := range st.ListTensors("model.layers.0.") {
			ti, _ := st.GetInfo(name)
			short := strings.TrimPrefix(name, "model.layers.0.")
			fmt.Printf("  %-40s %s %v\n", short, ti.Dtype, ti.Shape)
		}
	}
}

// === bench ===

func cmdBench() {
	fmt.Println("ai GPU Benchmark")
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

	var gpu mongoose.Engine

	if runtime.GOOS == "darwin" {
		if m := mongoose.NewMetal(); m != nil {
			fmt.Printf("  Metal: %s\n", m.Name())
			gpu = m
		}
	}
	if c := mongoose.NewCUDA(); c != nil {
		fmt.Printf("  CUDA:  %s\n", c.Name())
		if gpu == nil {
			gpu = c
		}
	}
	if w := mongoose.NewWebGPU(); w != nil {
		fmt.Printf("  Vulkan: %s\n", w.Name())
		if gpu == nil {
			gpu = w
		}
	}
	cpu := &mongoose.CPU{}
	fmt.Printf("  CPU:   %s (%.1f GFLOPS)\n", cpu.Name(), cpu.Benchmark())

	if gpu != nil {
		fmt.Println("\nScheduler calibration:")
		sched := mongoose.NewScheduler(gpu, cpu)
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
		fmt.Fprintln(os.Stderr, "Usage: ai export qdrant <url> <collection> [output.jsonl]")
		fmt.Fprintln(os.Stderr, "       ai export npy <input.npy> <output.jsonl>")
		os.Exit(1)
	}

	switch args[0] {
	case "qdrant":
		if len(args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: ai export qdrant <url> <collection> [output.jsonl]")
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
			fmt.Fprintln(os.Stderr, "Usage: ai export npy <input.npy> [output.jsonl]")
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
	if _, err := os.Stat(name); err == nil {
		return name
	}
	home, _ := os.UserHomeDir()
	dirs := []string{
		modelsDir,
		filepath.Join(home, ".ai", "models"),
	}
	for _, dir := range dirs {
		for _, suffix := range []string{"", "-hf", "-chat"} {
			p := filepath.Join(dir, name+suffix)
			if _, err := os.Stat(p); err == nil {
				return p
			}
		}
	}
	log.Fatalf("Model not found: %s\nTry: ai pull <org>/%s", name, name)
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

func geluNew(x float32) float32 {
	xf := float64(x)
	return float32(0.5 * xf * (1.0 + math.Tanh(math.Sqrt(2.0/math.Pi)*(xf+0.044715*xf*xf*xf))))
}

func applyRoPEPartial(x []float32, pos, headDim, rotaryDim, numHeads int, theta float32) {
	half := rotaryDim / 2
	for h := 0; h < numHeads; h++ {
		base := h * headDim
		for i := 0; i < half; i++ {
			freq := float32(1.0 / math.Pow(float64(theta), float64(2*i)/float64(rotaryDim)))
			val := float32(pos) * freq
			cos := float32(math.Cos(float64(val)))
			sin := float32(math.Sin(float64(val)))
			x0 := x[base+i]
			x1 := x[base+half+i]
			x[base+i] = x0*cos - x1*sin
			x[base+half+i] = x0*sin + x1*cos
		}
	}
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


func applyRoPESingle(x []float32, pos, headDim, numHeads int, theta float64) {
	half := headDim / 2
	for h := 0; h < numHeads; h++ {
		base := h * headDim
		for i := 0; i < half; i++ {
			freq := 1.0 / math.Pow(theta, float64(2*i)/float64(headDim))
			angle := float64(pos) * freq
			cos := float32(math.Cos(angle))
			sin := float32(math.Sin(angle))
			x0 := x[base+i]
			x1 := x[base+half+i]
			x[base+i] = x0*cos - x1*sin
			x[base+half+i] = x0*sin + x1*cos
		}
	}
}
