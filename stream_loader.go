package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math/rand"
	"os"
	"sort"
)

// StreamLoader reads NDJSON training data line-by-line with a shuffle buffer.
// Never loads the whole file into RAM — suitable for datasets exceeding DRAM.
//
// Usage:
//
//	loader, err := NewStreamLoader("trades.ndjson", StreamConfig{BufferSize: 100000, BatchSize: 512})
//	for {
//	    features, labels, err := loader.NextBatch()
//	    if err == io.EOF { break }
//	    // features: [batchSize * inputDim], labels: [batchSize]
//	}
//	loader.Reset() // next epoch
type StreamLoader struct {
	path  string
	cfg   StreamConfig
	file  *os.File
	scan  *bufio.Scanner
	buf   []sample
	keys  []string // sorted signal keys for consistent feature ordering
	rng   *rand.Rand
	epoch int
	total int64 // samples yielded this epoch
	eof   bool  // underlying file exhausted this epoch
}

type sample struct {
	features []float32
	label    float32
}

// StreamConfig controls the streaming data loader.
type StreamConfig struct {
	BufferSize int    // shuffle buffer capacity (default 100_000)
	BatchSize  int    // mini-batch size (default 512)
	LabelField string // JSON field for the label (default "profit")
	Seed       int64  // RNG seed for shuffle (default 42)
}

// gateRecord mirrors the NDJSON format from optimization-engine gate-data.
type gateRecord struct {
	Profit      float64            `json:"profit"`
	CloseReason string             `json:"close_reason"`
	Symbol      string             `json:"symbol"`
	Direction   string             `json:"direction"`
	SLDist      float64            `json:"sl_dist"`
	TPDist      float64            `json:"tp_dist"`
	RR          float64            `json:"rr"`
	Signals     map[string]float64 `json:"signals"`
}

// NewStreamLoader opens an NDJSON file and discovers the feature schema from the first line.
func NewStreamLoader(path string, cfg StreamConfig) (*StreamLoader, error) {
	if cfg.BufferSize == 0 {
		cfg.BufferSize = 100_000
	}
	if cfg.BatchSize == 0 {
		cfg.BatchSize = 512
	}
	if cfg.Seed == 0 {
		cfg.Seed = 42
	}

	// Discover signal keys from first line
	keys, err := discoverKeys(path)
	if err != nil {
		return nil, fmt.Errorf("stream_loader: discover keys: %w", err)
	}

	sl := &StreamLoader{
		path: path,
		cfg:  cfg,
		keys: keys,
		buf:  make([]sample, 0, cfg.BufferSize),
		rng:  rand.New(rand.NewSource(cfg.Seed)),
	}

	if err := sl.open(); err != nil {
		return nil, err
	}

	log.Printf("[stream] %s: %d signal features + 4 trade params = %d input dim, buffer=%d, batch=%d",
		path, len(keys), sl.InputDim(), cfg.BufferSize, cfg.BatchSize)

	// Initial fill
	sl.fill()

	return sl, nil
}

// InputDim returns the total feature vector size (signals + trade params).
func (sl *StreamLoader) InputDim() int {
	return len(sl.keys) + 4 // signals + direction + sl_dist + tp_dist + rr
}

// SignalKeys returns the ordered signal keys (needed for GATE binary export).
func (sl *StreamLoader) SignalKeys() []string {
	return sl.keys
}

// Epoch returns the current epoch number (0-based).
func (sl *StreamLoader) Epoch() int {
	return sl.epoch
}

// NextBatch returns the next mini-batch from the shuffle buffer.
// Returns io.EOF when the buffer is exhausted and the file is done for this epoch.
func (sl *StreamLoader) NextBatch() (features []float32, labels []float32, err error) {
	// Refill if buffer is below 25% and file isn't exhausted
	if len(sl.buf) < sl.cfg.BufferSize/4 && !sl.eof {
		sl.fill()
	}

	if len(sl.buf) == 0 {
		return nil, nil, io.EOF
	}

	bs := sl.cfg.BatchSize
	if bs > len(sl.buf) {
		bs = len(sl.buf)
	}

	dim := sl.InputDim()
	features = make([]float32, bs*dim)
	labels = make([]float32, bs)

	for i := 0; i < bs; i++ {
		// Fisher-Yates: swap random element to position i
		j := i + sl.rng.Intn(len(sl.buf)-i)
		sl.buf[i], sl.buf[j] = sl.buf[j], sl.buf[i]

		copy(features[i*dim:], sl.buf[i].features)
		labels[i] = sl.buf[i].label
	}

	// Remove consumed samples
	sl.buf = sl.buf[bs:]
	sl.total += int64(bs)

	return features, labels, nil
}

// Reset reopens the file for a new epoch.
func (sl *StreamLoader) Reset() error {
	sl.Close()
	sl.epoch++
	sl.total = 0
	sl.eof = false
	if err := sl.open(); err != nil {
		return err
	}
	sl.buf = sl.buf[:0]
	sl.fill()
	return nil
}

// Close releases the file handle.
func (sl *StreamLoader) Close() {
	if sl.file != nil {
		sl.file.Close()
		sl.file = nil
	}
}

// Stats returns current loader state for logging.
func (sl *StreamLoader) Stats() (buffered int, yielded int64, epoch int) {
	return len(sl.buf), sl.total, sl.epoch
}

// ComputeNormalization scans the entire file to compute per-feature mean and std.
// This is a full pass — call once before training, not per epoch.
func (sl *StreamLoader) ComputeNormalization() (mean, std []float32) {
	dim := sl.InputDim()
	mean = make([]float32, dim)
	std = make([]float32, dim)

	f, err := os.Open(sl.path)
	if err != nil {
		log.Printf("WARN stream normalization: %v", err)
		for i := range std {
			std[i] = 1
		}
		return
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 64*1024), 64*1024)

	// Welford's online algorithm for numerical stability
	n := int64(0)
	m2 := make([]float64, dim)
	dmean := make([]float64, dim)

	for scanner.Scan() {
		s, ok := sl.parseLine(scanner.Bytes())
		if !ok {
			continue
		}
		n++
		for i, v := range s.features {
			delta := float64(v) - dmean[i]
			dmean[i] += delta / float64(n)
			delta2 := float64(v) - dmean[i]
			m2[i] += delta * delta2
		}
	}

	if n < 2 {
		for i := range std {
			std[i] = 1
		}
		return
	}

	for i := range mean {
		mean[i] = float32(dmean[i])
		variance := m2[i] / float64(n-1)
		if variance < 1e-12 {
			std[i] = 1
		} else {
			std[i] = float32(variance)
			// sqrt
			s := std[i]
			for j := 0; j < 10; j++ {
				s = (s + std[i]/s) / 2
			}
			std[i] = s
		}
	}

	log.Printf("[stream] normalization computed over %d samples", n)
	return
}

// --- internals ---

func (sl *StreamLoader) open() error {
	f, err := os.Open(sl.path)
	if err != nil {
		return fmt.Errorf("stream_loader: open %s: %w", sl.path, err)
	}
	sl.file = f
	sl.scan = bufio.NewScanner(f)
	sl.scan.Buffer(make([]byte, 64*1024), 64*1024)
	sl.eof = false
	return nil
}

func (sl *StreamLoader) fill() {
	target := sl.cfg.BufferSize - len(sl.buf)
	added := 0

	for added < target && sl.scan.Scan() {
		s, ok := sl.parseLine(sl.scan.Bytes())
		if !ok {
			continue
		}
		sl.buf = append(sl.buf, s)
		added++
	}

	if sl.scan.Err() != nil || added < target {
		sl.eof = true
	}
}

func (sl *StreamLoader) parseLine(line []byte) (sample, bool) {
	var rec gateRecord
	if err := json.Unmarshal(line, &rec); err != nil {
		return sample{}, false
	}

	// Skip manual/friday closes — not signal-driven outcomes
	if rec.CloseReason == "MANUAL_CLOSE" || rec.CloseReason == "FRIDAY_CLOSE" {
		return sample{}, false
	}

	dim := sl.InputDim()
	features := make([]float32, dim)

	// Signal features in key order
	for i, key := range sl.keys {
		if v, ok := rec.Signals[key]; ok {
			features[i] = float32(v)
		}
	}

	// Trade params (appended after signals)
	base := len(sl.keys)
	if rec.Direction == "long" || rec.Direction == "LONG" || rec.Direction == "1" {
		features[base] = 1.0
	}
	features[base+1] = float32(rec.SLDist)
	features[base+2] = float32(rec.TPDist)
	features[base+3] = float32(rec.RR)

	// Label: profit > 0 → 1.0
	label := float32(0)
	if rec.Profit > 0 {
		label = 1.0
	}

	return sample{features: features, label: label}, true
}

// discoverKeys reads the first line of an NDJSON file to extract sorted signal keys.
func discoverKeys(path string) ([]string, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 64*1024), 64*1024)

	if !scanner.Scan() {
		return nil, fmt.Errorf("empty file: %s", path)
	}

	var rec gateRecord
	if err := json.Unmarshal(scanner.Bytes(), &rec); err != nil {
		return nil, fmt.Errorf("parse first line: %w", err)
	}

	keys := make([]string, 0, len(rec.Signals))
	for k := range rec.Signals {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys, nil
}
