package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/tensorwire/mongoose"
)

// cmdTrainMLP: ai train --mlp data=<file> [options]
//
// Trains an MLP classifier on tabular data (NDJSON or raw binary).
// Uses GPU-accelerated forward/backward when available.
//
// Options:
//   data=<file>       Training data (NDJSON with "features" + "label" fields, or .bin)
//   output=<file>     Output model path (default: gate-model.bin)
//   hidden=512,256,128  Hidden layer sizes (default: 512,256,128)
//   epochs=30         Number of epochs (default: 30)
//   lr=0.0003         Learning rate (default: 0.0003)
//   wd=0.0005         Weight decay (default: 0.0005)
//   batch=512         Batch size (default: 512)
//   dropout=0.2       Dropout rate (default: 0.2)
//   split=0.9         Train/val split ratio (default: 0.9)
func cmdTrainMLP(args map[string]string) {
	dataPath := args["data"]
	if dataPath == "" {
		fmt.Fprintln(os.Stderr, "Usage: ai train --mlp data=<file> [output=model.bin] [hidden=512,256,128] [epochs=30] [lr=0.0003]")
		os.Exit(1)
	}

	outputPath := kvString(args, "output", "gate-model.bin")
	hiddenStr := kvString(args, "hidden", "512,256,128")
	epochs := kvInt(args, "epochs", 30)
	lr := kvFloat(args, "lr", 0.0003)
	wd := kvFloat(args, "wd", 0.0005)
	batchSize := kvInt(args, "batch", 512)
	dropout := kvFloat(args, "dropout", 0.2)
	splitRatio := kvFloat(args, "split", 0.9)

	// Parse hidden layer sizes
	var hidden []int
	for _, s := range strings.Split(hiddenStr, ",") {
		var v int
		fmt.Sscanf(strings.TrimSpace(s), "%d", &v)
		if v > 0 {
			hidden = append(hidden, v)
		}
	}

	// Load data
	features, labels, featureNames := loadMLPData(dataPath)
	n := len(labels)
	nFeatures := len(features) / n

	fmt.Println("ai train --mlp")
	fmt.Printf("  data:     %s (%d samples, %d features)\n", dataPath, n, nFeatures)
	fmt.Printf("  hidden:   %v\n", hidden)
	fmt.Printf("  epochs:   %d\n", epochs)
	fmt.Printf("  lr:       %.4f\n", lr)
	fmt.Printf("  wd:       %.4f\n", wd)
	fmt.Printf("  batch:    %d\n", batchSize)
	fmt.Printf("  dropout:  %.2f\n", dropout)
	fmt.Printf("  output:   %s\n", outputPath)

	// Normalize features
	mean := make([]float32, nFeatures)
	std := make([]float32, nFeatures)
	split := int(float64(n) * splitRatio)

	for d := 0; d < nFeatures; d++ {
		var sum float64
		for i := 0; i < split; i++ {
			sum += float64(features[i*nFeatures+d])
		}
		mean[d] = float32(sum / float64(split))
	}
	for d := 0; d < nFeatures; d++ {
		var sum float64
		for i := 0; i < split; i++ {
			diff := float64(features[i*nFeatures+d]) - float64(mean[d])
			sum += diff * diff
		}
		std[d] = float32(math.Sqrt(sum / float64(split)))
		if std[d] == 0 {
			std[d] = 1
		}
	}
	for i := 0; i < n; i++ {
		for d := 0; d < nFeatures; d++ {
			features[i*nFeatures+d] = (features[i*nFeatures+d] - mean[d]) / std[d]
		}
	}

	// Pos weight for class imbalance
	var posCount float32
	for i := 0; i < split; i++ {
		posCount += labels[i]
	}
	posWeight := float32(split-int(posCount)) / posCount
	fmt.Printf("  pos_weight: %.2f (%.1f%% positive)\n", posWeight, 100*posCount/float32(split))

	// Create MLP
	dims := append([]int{nFeatures}, hidden...)
	dims = append(dims, 1)
	mlp := mongoose.NewMLP(dims, mongoose.MLPConfig{
		Activation: "relu",
		Dropout:    float32(dropout),
		BatchNorm:  true,
		Sigmoid:    true,
	})
	fmt.Printf("  params:   %d\n\n", mlp.ParamCount())

	// GPU setup
	eng := selectEngine("auto")
	te := mongoose.AsTensorEngine(eng)
	cuda, _ := eng.(*mongoose.CUDA)
	metal, _ := eng.(*mongoose.Metal)
	var fused *mongoose.MLPFused
	var metalMLP *mongoose.MLPMetal
	if te != nil {
		mongoose.LoadKernels()
		if cuda != nil {
			fused = mongoose.NewMLPFused(cuda, mlp, batchSize)
			if fused != nil {
				fmt.Printf("  engine:   %s (fused single-kernel GPU training)\n", eng.Name())
			} else {
				mlp.ToGPU(te)
				if mongoose.HasMLPKernels() {
					fmt.Printf("  engine:   %s (GPU kernels)\n", eng.Name())
				} else {
					fmt.Printf("  engine:   %s (GPU matmul, CPU bias/BN/activation)\n", eng.Name())
				}
			}
		} else if metal != nil {
			metalMLP = mongoose.NewMLPMetal(metal, mlp, batchSize)
			if metalMLP != nil {
				fmt.Printf("  engine:   %s (Metal fused MLP training)\n", eng.Name())
			} else {
				mlp.ToGPU(te)
				fmt.Printf("  engine:   %s (GPU matmul)\n", eng.Name())
			}
		} else {
			mlp.ToGPU(te)
			fmt.Printf("  engine:   %s (GPU matmul)\n", eng.Name())
		}
	} else {
		fmt.Printf("  engine:   CPU\n")
	}

	opt := mongoose.NewMLPOptimizer(float32(lr), float32(wd), epochs)
	rng := rand.New(rand.NewSource(time.Now().UnixNano()))

	// Training loop
	trainX := features[:split*nFeatures]
	trainY := labels[:split]
	valX := features[split*nFeatures:]
	valY := labels[split:]

	nBatches := split / batchSize
	batchX := make([]float32, batchSize*nFeatures)
	batchY := make([]float32, batchSize)

	var bestValLoss float32 = 999
	var bestAcc float32
	var bestEpoch int

	fmt.Println("Training...")

	t0 := time.Now()

	for epoch := 0; epoch < epochs; epoch++ {
		cosLR := float32(float64(lr) * 0.5 * (1.0 + math.Cos(math.Pi*float64(epoch)/float64(epochs))))

		// Shuffle training indices
		indices := rng.Perm(split)

		var epochLoss float64
		for b := 0; b < nBatches; b++ {
			for i := 0; i < batchSize; i++ {
				idx := indices[b*batchSize+i]
				copy(batchX[i*nFeatures:(i+1)*nFeatures], trainX[idx*nFeatures:(idx+1)*nFeatures])
				batchY[i] = trainY[idx]
			}

			if fused != nil {
				fused.UploadBatch(batchX, batchY)
				loss := fused.TrainStep(cosLR)
				epochLoss += float64(loss)
			} else if metalMLP != nil {
				metalMLP.UploadBatch(batchX, batchY)
				loss := metalMLP.TrainStep(cosLR)
				epochLoss += float64(loss)
			} else if te != nil {
				inputT := te.FromHost(batchX, []int{batchSize, nFeatures})
				outT := mlp.ForwardGPU(inputT, batchSize, true)
				logits := te.ToHost(outT)
				te.Release(outT)
				te.Release(inputT)

				loss, grad := mlp.BCEWithLogitsLoss(logits, batchY, posWeight)
				epochLoss += float64(loss)

				mlp.BackwardGPU(grad, batchSize, true)
				params, grads := mlp.Params()
				opt.Step(params, grads, epoch)
				mlp.SyncWeightsToGPU()
			} else {
				logits := mlp.ForwardLogits(batchX, batchSize, true)
				loss, grad := mlp.BCEWithLogitsLoss(logits, batchY, posWeight)
				epochLoss += float64(loss)

				mlp.BackwardFromLogits(grad, batchSize)
				params, grads := mlp.Params()
				opt.Step(params, grads, epoch)
			}
		}

		// Validation only on reporting epochs (avoid GPU→CPU sync every epoch)
		reportEpoch := (epoch+1)%5 == 0 || epoch == 0 || epoch == epochs-1
		if reportEpoch {
			if fused != nil {
				fused.DownloadWeights()
			} else if metalMLP != nil {
				metalMLP.DownloadWeights()
			}

			valPreds := mlp.ForwardLogits(valX, len(valY), false)
			valLoss, _ := mlp.BCEWithLogitsLoss(valPreds, valY, 0)

			correct := 0
			for i, p := range valPreds {
				sig := 1.0 / (1.0 + math.Exp(-float64(p)))
				pred := 0
				if sig > 0.5 {
					pred = 1
				}
				if float32(pred) == valY[i] {
					correct++
				}
			}
			acc := float32(correct) / float32(len(valY))

			if valLoss < bestValLoss {
				bestValLoss = valLoss
				bestAcc = acc
				bestEpoch = epoch
			}

			elapsed := time.Since(t0)
			fmt.Printf("  epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.3f  best_ep=%d  (%.0fs)\n",
				epoch+1, epochs, epochLoss/float64(nBatches), valLoss, acc, bestEpoch+1, elapsed.Seconds())
		}
	}

	elapsed := time.Since(t0)
	fmt.Printf("\nTraining complete: %d epochs in %.1fs (%.1f epochs/s)\n", epochs, elapsed.Seconds(), float64(epochs)/elapsed.Seconds())
	fmt.Printf("Best: epoch %d, val_loss=%.4f, val_acc=%.3f\n", bestEpoch+1, bestValLoss, bestAcc)

	// Download weights from GPU
	if fused != nil {
		fused.DownloadWeights()
		fused.Destroy()
	} else if metalMLP != nil {
		metalMLP.DownloadWeights()
		metalMLP.Destroy()
	} else if te != nil {
		mlp.ToCPU()
	}

	// Export
	exportMLPGate(mlp, mean, std, featureNames, outputPath)
	fmt.Printf("Model saved to %s\n", outputPath)
}

func loadMLPData(path string) (features []float32, labels []float32, featureNames []string) {
	ext := strings.ToLower(filepath.Ext(path))

	if ext == ".bin" {
		return loadMLPBinary(path)
	}

	// NDJSON format: {"features": {...}, "label": 0/1}
	f, err := os.Open(path)
	if err != nil {
		log.Fatalf("open data: %v", err)
	}
	defer f.Close()

	data, _ := io.ReadAll(f)
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")

	// First pass: determine feature names from first record
	var first map[string]interface{}
	json.Unmarshal([]byte(lines[0]), &first)

	if feats, ok := first["features"].(map[string]interface{}); ok {
		for k := range feats {
			featureNames = append(featureNames, k)
		}
	} else if feats, ok := first["signals"].(map[string]interface{}); ok {
		for k := range feats {
			featureNames = append(featureNames, k)
		}
	}
	// Sort for deterministic order
	sortStrings(featureNames)
	nFeatures := len(featureNames)

	for _, line := range lines {
		if strings.TrimSpace(line) == "" {
			continue
		}
		var rec map[string]interface{}
		json.Unmarshal([]byte(line), &rec)

		feats := rec["features"]
		if feats == nil {
			feats = rec["signals"]
		}
		fm, ok := feats.(map[string]interface{})
		if !ok {
			continue
		}

		for _, name := range featureNames {
			v, _ := fm[name].(float64)
			if math.IsNaN(v) || math.IsInf(v, 0) {
				v = 0
			}
			features = append(features, float32(v))
		}

		label := float32(0)
		if l, ok := rec["label"].(float64); ok {
			label = float32(l)
		} else if l, ok := rec["profit"].(float64); ok && l > 0 {
			label = 1
		}
		labels = append(labels, label)
	}

	if len(featureNames) == 0 || len(labels) == 0 {
		log.Fatalf("no valid records in %s (need NDJSON with 'features'/'signals' + 'label'/'profit')", path)
	}

	_ = nFeatures
	return
}

func loadMLPBinary(path string) (features []float32, labels []float32, featureNames []string) {
	f, err := os.Open(path)
	if err != nil {
		log.Fatalf("open binary: %v", err)
	}
	defer f.Close()

	var magic [4]byte
	f.Read(magic[:])
	if string(magic[:]) != "MLPD" {
		log.Fatalf("bad magic: %q (expected MLPD)", magic)
	}

	var nSamples, nFeatures uint32
	binary.Read(f, binary.LittleEndian, &nSamples)
	binary.Read(f, binary.LittleEndian, &nFeatures)

	features = make([]float32, nSamples*nFeatures)
	labels = make([]float32, nSamples)
	binary.Read(f, binary.LittleEndian, features)
	binary.Read(f, binary.LittleEndian, labels)

	for i := 0; i < int(nFeatures); i++ {
		featureNames = append(featureNames, fmt.Sprintf("f%d", i))
	}
	return
}

func exportMLPGate(mlp *mongoose.MLP, mean, std []float32, featureNames []string, path string) {
	f, err := os.Create(path)
	if err != nil {
		log.Fatalf("create output: %v", err)
	}
	defer f.Close()

	nFeatures := len(mean)
	nLinear := 0
	nBN := 0
	for _, l := range mlp.Layers {
		nLinear++
		if l.BNGamma != nil {
			nBN++
		}
	}

	f.Write([]byte("GATE"))
	binary.Write(f, binary.LittleEndian, uint32(2)) // version 2
	binary.Write(f, binary.LittleEndian, uint32(nLinear))

	for _, l := range mlp.Layers {
		binary.Write(f, binary.LittleEndian, uint32(l.InDim))
		binary.Write(f, binary.LittleEndian, uint32(l.OutDim))
		flags := uint32(0)
		if l.BNGamma != nil {
			flags |= 1
		}
		binary.Write(f, binary.LittleEndian, flags)
		binary.Write(f, binary.LittleEndian, l.W)
		binary.Write(f, binary.LittleEndian, l.B)
		if l.BNGamma != nil {
			binary.Write(f, binary.LittleEndian, l.BNGamma)
			binary.Write(f, binary.LittleEndian, l.BNBeta)
			binary.Write(f, binary.LittleEndian, l.BNMean)
			binary.Write(f, binary.LittleEndian, l.BNVar)
		}
	}

	binary.Write(f, binary.LittleEndian, uint32(nFeatures))
	binary.Write(f, binary.LittleEndian, mean)
	binary.Write(f, binary.LittleEndian, std)

	for _, name := range featureNames {
		enc := []byte(name)
		binary.Write(f, binary.LittleEndian, uint16(len(enc)))
		f.Write(enc)
	}
}

func sortStrings(s []string) {
	for i := 0; i < len(s); i++ {
		for j := i + 1; j < len(s); j++ {
			if s[j] < s[i] {
				s[i], s[j] = s[j], s[i]
			}
		}
	}
}

func kvString(args map[string]string, key, def string) string {
	if v, ok := args[key]; ok {
		return v
	}
	return def
}
