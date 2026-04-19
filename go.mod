module github.com/open-ai-org/tesseract

go 1.25.0

require (
	github.com/open-ai-org/gguf v0.0.0
	github.com/open-ai-org/mongoose v0.0.0
	github.com/open-ai-org/tokenizer v0.0.0
)

require (
	github.com/go-webgpu/goffi v0.5.0 // indirect
	github.com/gogpu/gputypes v0.4.0 // indirect
	github.com/gogpu/naga v0.17.3 // indirect
	github.com/gogpu/wgpu v0.24.7 // indirect
	golang.org/x/sys v0.43.0 // indirect
)

replace (
	github.com/open-ai-org/gguf => ../gguf
	github.com/open-ai-org/helix => ../helix
	github.com/open-ai-org/mongoose => ../mongoose
	github.com/open-ai-org/needle => ../needle
	github.com/open-ai-org/tokenizer => ../tokenizer
)
