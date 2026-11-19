// This local example writes a configuration bundle. It never starts training.
package main

import (
	"bytes"
	"crypto/sha256"
	"embed"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/oam-dev/kubevela/pkg/ai/training"
)

//go:embed config.json alpaca.json
var fixtures embed.FS

func main() {
	out := flag.String("out", "", "new host directory for the bundle (required)")
	datasetDir := flag.String("dataset-dir", "/app/data/kubevela-sft-smoke-20261119", "prepared Linux container data directory")
	outputDir := flag.String("output-dir", "/app/saves/kubevela-sft-smoke-20261119-02", "new Linux container output directory")
	flag.Parse()
	if err := generate(*out, *datasetDir, *outputDir); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func generate(out, datasetDir, outputDir string) error {
	if out == "" {
		return fmt.Errorf("-out must name a new host directory")
	}
	raw, err := fixtures.ReadFile("config.json")
	if err != nil {
		return err
	}
	c, err := training.DecodeJSON(bytes.NewReader(raw))
	if err != nil {
		return err
	}
	data, err := fixtures.ReadFile("alpaca.json")
	if err != nil {
		return err
	}
	if fmt.Sprintf("sha256:%x", sha256.Sum256(data)) != c.Dataset.Version {
		return fmt.Errorf("sample dataset digest does not match config.json")
	}
	bundle, err := training.GenerateLLaMAFactory(c, training.LLaMAFactoryPaths{
		DatasetDir: datasetDir, TrainFile: "alpaca.json", OutputDir: outputDir,
	})
	if err != nil {
		return err
	}
	// Mkdir deliberately refuses an existing directory to preserve old bundles.
	if err := os.Mkdir(out, 0755); err != nil {
		return err
	}
	for name, content := range map[string][]byte{
		"config.json": raw, "alpaca.json": data,
		"train.yaml": bundle.TrainYAML, "dataset_info.json": bundle.DatasetInfoJSON,
	} {
		if err := os.WriteFile(filepath.Join(out, name), content, 0644); err != nil {
			return err
		}
	}
	fmt.Println("Generated configuration bundle:", out)
	return nil
}
