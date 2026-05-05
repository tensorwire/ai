package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/tensorwire/mongoose"
)

// cmdTrainMLP trains a generic MLP on NDJSON tabular data with streaming.
//
//	ai train data=trades.ndjson arch=mlp --stream
//	ai train data=trades.ndjson arch=mlp --stream layers=512,256,128 lr=3e-4 dropout=0.2 epochs=300
func cmdTrainMLP(args map[string]string) {
	dataPath := args["data"]
	if dataPath == "" {
		log.Fatal("Usage: ai train data=<file.ndjson> arch=mlp --stream")
	}

	// Parse hyperparameters
	layerStr := args["layers"]
	if layerStr == "" {
		layerStr = "512,256,128"
	}
	hiddenDims := parseIntList(layerStr)

	lr := kvFloat(args, "lr", 3e-4)
	dropout := float32(kvFloat(args, "dropout", 0.2))
	weightDecay := float32(kvFloat(args, "weight_decay", 5e-4))
	labelSmooth := float32(kvFloat(args, "label_smooth", 0.05))
	epochs := kvInt(args, "epochs", 100)
	batchSize := kvInt(args, "batch", 512)
	bufferSize := kvInt(args, "buffer", 100_000)
	logEvery := kvInt(args, "log", 100)
	valSplit := kvFloat(args, "val", 0.1)
	patience := kvInt(args, "patience", 10)

	output := args["output"]
	if output == "" {
		ext := filepath.Ext(dataPath)
		output = strings.TrimSuffix(dataPath, ext) + ".gate.bin"
	}

	activation := args["activation"]
	if activation == "" {
		activation = "relu"
	}

	// Open streaming loader
	loader, err := NewStreamLoader(dataPath, StreamConfig{
		BufferSize: bufferSize,
		BatchSize:  batchSize,
	})
	if err != nil {
		log.Fatalf("open data: %v", err)
	}
	defer loader.Close()

	inputDim := loader.InputDim()

	// Build MLP dimensions: [inputDim, hidden..., 1]
	dims := make([]int, 0, len(hiddenDims)+2)
	dims = append(dims, inputDim)
	dims = append(dims, hiddenDims...)
	dims = append(dims, 1)

	mlp := mongoose.NewMLP(dims, mongoose.MLPConfig{
		Activation:  activation,
		Dropout:     dropout,
		BatchNorm:   true,
		Sigmoid:     true,
		LabelSmooth: labelSmooth,
		WeightDecay: weightDecay,
	})

	log.Printf("[mlp] arch: %v (%d params)", dims, mlp.ParamCount())
	log.Printf("[mlp] lr=%.1e dropout=%.2f wd=%.1e smooth=%.2f epochs=%d batch=%d",
		lr, dropout, weightDecay, labelSmooth, epochs, batchSize)

	// Compute normalization stats (full pass over file)
	log.Println("[mlp] computing normalization stats (full scan)...")
	mean, std := loader.ComputeNormalization()

	// AdamW state
	params, _ := mlp.Params()
	m := make([][]float32, len(params)) // first moment
	v := make([][]float32, len(params)) // second moment
	for i, p := range params {
		m[i] = make([]float32, len(p))
		v[i] = make([]float32, len(p))
	}

	// Validation buffer — fill from first val_split fraction of buffer
	var valFeatures, valLabels []float32
	if valSplit > 0 {
		log.Printf("[mlp] loading validation set (%.0f%% of first buffer)...", valSplit*100)
		valFeatures, valLabels = loadValidation(loader, valSplit, mean, std)
		log.Printf("[mlp] validation set: %d samples", len(valLabels))
		loader.Reset()
	}

	bestValLoss := float32(math.MaxFloat32)
	bestParams := snapshotParams(mlp)
	staleEpochs := 0
	step := 0

	start := time.Now()

	for epoch := 0; epoch < epochs; epoch++ {
		if epoch > 0 {
			loader.Reset()
		}

		epochLoss := float64(0)
		epochSamples := 0
		epochCorrect := 0

		for {
			features, labels, err := loader.NextBatch()
			if err == io.EOF {
				break
			}
			if err != nil {
				log.Fatalf("batch read: %v", err)
			}

			bs := len(labels)

			// Normalize features
			normalize(features, mean, std, inputDim)

			// Forward
			preds := mlp.Forward(features, bs, true)

			// Loss
			loss, grad := mlp.BCELoss(preds, labels)

			// Backward
			mlp.Backward(grad, bs)

			// AdamW step
			step++
			adamWStep(mlp, m, v, lr, weightDecay, step)

			epochLoss += float64(loss) * float64(bs)
			epochSamples += bs

			// Accuracy
			for i, p := range preds {
				pred := 0
				if p >= 0.5 {
					pred = 1
				}
				actual := 0
				if labels[i] >= 0.5 {
					actual = 1
				}
				if pred == actual {
					epochCorrect++
				}
			}

			if step%logEvery == 0 {
				acc := float64(epochCorrect) / float64(epochSamples) * 100
				_, yielded, _ := loader.Stats()
				log.Printf("[mlp] epoch %d step %d | loss=%.4f acc=%.1f%% | %.0f samples/s | buf=%d yielded=%d",
					epoch+1, step, loss, acc, float64(epochSamples)/time.Since(start).Seconds(), len(loader.buf), yielded)
			}
		}

		avgLoss := float32(epochLoss / float64(epochSamples))
		acc := float64(epochCorrect) / float64(epochSamples) * 100

		// Validation
		valLoss := float32(0)
		valAcc := float64(0)
		if len(valLabels) > 0 {
			valLoss, valAcc = evaluate(mlp, valFeatures, valLabels, inputDim)
		}

		log.Printf("[mlp] epoch %d/%d | train loss=%.4f acc=%.1f%% | val loss=%.4f acc=%.1f%% | %s",
			epoch+1, epochs, avgLoss, acc, valLoss, valAcc, time.Since(start).Round(time.Second))

		// Early stopping
		if valLoss < bestValLoss {
			bestValLoss = valLoss
			bestParams = snapshotParams(mlp)
			staleEpochs = 0
		} else {
			staleEpochs++
			if patience > 0 && staleEpochs >= patience {
				log.Printf("[mlp] early stopping: no improvement for %d epochs", patience)
				break
			}
		}
	}

	// Restore best params
	restoreParams(mlp, bestParams)

	// Final evaluation
	if len(valLabels) > 0 {
		finalLoss, finalAcc := evaluate(mlp, valFeatures, valLabels, inputDim)
		log.Printf("[mlp] final (best checkpoint) | val loss=%.4f acc=%.1f%%", finalLoss, finalAcc)
	}

	// Export GATE binary
	exportGATE(output, mlp, loader.SignalKeys(), mean, std)
	log.Printf("[mlp] model saved: %s (%d params, %d input features)",
		output, mlp.ParamCount(), inputDim)

	elapsed := time.Since(start).Round(time.Second)
	log.Printf("[mlp] training complete in %s (%d steps)", elapsed, step)
}

// --- helpers ---

func parseIntList(s string) []int {
	parts := strings.Split(s, ",")
	result := make([]int, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		n, err := strconv.Atoi(p)
		if err != nil {
			log.Fatalf("invalid layer size %q: %v", p, err)
		}
		result = append(result, n)
	}
	return result
}

func normalize(features, mean, std []float32, dim int) {
	n := len(features) / dim
	for i := 0; i < n; i++ {
		for j := 0; j < dim; j++ {
			s := std[j]
			if s == 0 {
				s = 1
			}
			features[i*dim+j] = (features[i*dim+j] - mean[j]) / s
		}
	}
}

func adamWStep(mlp *mongoose.MLP, mState, vState [][]float32, lr float64, wd float32, step int) {
	const beta1 = 0.9
	const beta2 = 0.999
	const eps = 1e-8

	bc1 := 1.0 - math.Pow(beta1, float64(step))
	bc2 := 1.0 - math.Pow(beta2, float64(step))

	params, grads := mlp.Params()

	for i := range params {
		for j := range params[i] {
			g := grads[i][j]
			mState[i][j] = beta1*mState[i][j] + (1-beta1)*g
			vState[i][j] = beta2*vState[i][j] + (1-beta2)*g*g
			mHat := float64(mState[i][j]) / bc1
			vHat := float64(vState[i][j]) / bc2
			params[i][j] -= float32(lr * mHat / (math.Sqrt(vHat) + eps))
			// Weight decay (decoupled)
			params[i][j] -= float32(lr) * wd * params[i][j]
		}
	}
}

func evaluate(mlp *mongoose.MLP, features, labels []float32, inputDim int) (float32, float64) {
	bs := len(labels)
	preds := mlp.Forward(features, bs, false)
	loss, _ := mlp.BCELoss(preds, labels)
	correct := 0
	for i, p := range preds {
		pred := 0
		if p >= 0.5 {
			pred = 1
		}
		actual := 0
		if labels[i] >= 0.5 {
			actual = 1
		}
		if pred == actual {
			correct++
		}
	}
	return loss, float64(correct) / float64(bs) * 100
}

func loadValidation(loader *StreamLoader, split float64, mean, std []float32) ([]float32, []float32) {
	dim := loader.InputDim()
	nVal := int(float64(loader.cfg.BufferSize) * split)
	if nVal < 100 {
		nVal = 100
	}

	var allFeatures []float32
	var allLabels []float32

	collected := 0
	for collected < nVal {
		features, labels, err := loader.NextBatch()
		if err == io.EOF {
			break
		}
		if err != nil {
			break
		}
		normalize(features, mean, std, dim)
		allFeatures = append(allFeatures, features...)
		allLabels = append(allLabels, labels...)
		collected += len(labels)
	}

	if collected > nVal {
		allFeatures = allFeatures[:nVal*dim]
		allLabels = allLabels[:nVal]
	}

	return allFeatures, allLabels
}

func snapshotParams(mlp *mongoose.MLP) [][]float32 {
	params, _ := mlp.Params()
	snap := make([][]float32, len(params))
	for i, p := range params {
		snap[i] = make([]float32, len(p))
		copy(snap[i], p)
	}
	return snap
}

func restoreParams(mlp *mongoose.MLP, snap [][]float32) {
	params, _ := mlp.Params()
	for i := range params {
		copy(params[i], snap[i])
	}
}

// exportGATE writes the trained MLP to the GATE binary format that common-lib/gate loads.
func exportGATE(path string, mlp *mongoose.MLP, signalKeys []string, mean, std []float32) {
	f, err := os.Create(path)
	if err != nil {
		log.Fatalf("create %s: %v", path, err)
	}
	defer f.Close()

	inputDim := mlp.Layers[0].InDim
	nLayers := len(mlp.Layers)
	nBN := 0
	for _, l := range mlp.Layers {
		if l.BNGamma != nil {
			nBN++
		}
	}

	// Magic
	binary.Write(f, binary.LittleEndian, [4]byte{'G', 'A', 'T', 'E'})

	// Header
	binary.Write(f, binary.LittleEndian, uint32(1))               // version
	binary.Write(f, binary.LittleEndian, uint32(inputDim))        // input_dim
	binary.Write(f, binary.LittleEndian, uint32(len(signalKeys))) // n_signal_keys
	binary.Write(f, binary.LittleEndian, uint32(nLayers))         // n_layers
	binary.Write(f, binary.LittleEndian, uint32(nBN))             // n_bn

	// Normalization params
	binary.Write(f, binary.LittleEndian, mean)
	binary.Write(f, binary.LittleEndian, std)

	// Signal keys
	for _, key := range signalKeys {
		binary.Write(f, binary.LittleEndian, uint16(len(key)))
		f.Write([]byte(key))
	}

	// Linear layers
	for _, l := range mlp.Layers {
		binary.Write(f, binary.LittleEndian, uint32(l.OutDim)) // rows
		binary.Write(f, binary.LittleEndian, uint32(l.InDim))  // cols
		binary.Write(f, binary.LittleEndian, l.W)
		binary.Write(f, binary.LittleEndian, l.B)
	}

	// BatchNorm layers
	for _, l := range mlp.Layers {
		if l.BNGamma == nil {
			continue
		}
		binary.Write(f, binary.LittleEndian, uint32(l.OutDim))
		binary.Write(f, binary.LittleEndian, l.BNGamma)
		binary.Write(f, binary.LittleEndian, l.BNBeta)
		binary.Write(f, binary.LittleEndian, l.BNMean)
		binary.Write(f, binary.LittleEndian, l.BNVar)
	}

	fi, _ := f.Stat()
	log.Printf("[gate] exported %s (%s)", path, formatBytes(int(fi.Size())))
}
