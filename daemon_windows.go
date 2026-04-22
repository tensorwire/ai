//go:build windows

package main

import "os/exec"

func setSysProcAttr(cmd *exec.Cmd) {
	// Windows doesn't have Setsid — process is already detached via cmd.Start()
}
