// Copyright ©2013 The bíogo Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package norm_test

import (
	"code.google.com/p/biogo.rnaseq/norm"

	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
)

func random(n int, m, s float64) []float64 {
	v := make([]float64, n)
	for i := range v {
		vi := rand.NormFloat64()*s + m
		if vi > 0 {
			v[i] = math.Floor(vi)
		}
	}
	return v
}

func constrain(b, a []float64) {
	var sa, sb float64
	for _, v := range a {
		sa += v
	}
	for _, v := range b {
		sb += v
	}
	f := sa / sb
	for i, v := range b {
		b[i] = math.Floor(v*f + 0.5)
	}
}

func mean(a []float64) float64 {
	var s float64
	for _, v := range a {
		s += v
	}
	return s / float64(len(a))
}

func scale(a []float64, f float64) {
	for i := range a {
		a[i] *= f
	}
}

func Example_1() {
	data := [][]float64{
		{1, 3, 5, 0, 6, 9},
		{2, 4, 6, 0, 8, 10},
	}
	fmt.Println(norm.TMM(data, -1, 0.3, 0.05, -1e10, true))
	fmt.Println(norm.Quantile(data, 0.75))
	fmt.Println(norm.RelativeLog(data))

	// Output:
	// [0.9858731252345716 1.014329303034878] <nil>
	// [0.9682458365518544 1.0327955589886446] <nil>
	// [0.9682458365518544 1.0327955589886444] <nil>
}

func Example_2() {
	rand.Seed(1)

	// Make two 10,000 long sets of counts normally distributed
	// around 10 with a standard deviation of 3.
	a := random(1e4, 10, 3)
	b := random(1e4, 10, 3)

	// Replace the first 100 counts in b with counts normally
	// distributed around 100 with a standard deviation of 1.
	copy(b, random(1e2, 100, 1))

	// Constrain b such that the sum of all values in b is
	// approximately the same as for a.
	constrain(b, a)

	fmt.Println("Before normalisation:")
	fmt.Printf("All:    mean(a) = %.1f, mean(b) = %.1f\n", mean(a), mean(b))
	fmt.Printf("Common: mean(a) = %.1f, mean(b) = %.1f\n", mean(a[1e2:]), mean(b[1e2:]))
	fmt.Printf("Up:     mean(a) = %.1f, mean(b) = %.1f\n\n", mean(a[:1e2]), mean(b[:1e2]))

	f, err := norm.TMM([][]float64{a, b}, -1, 0.3, 0.05, -1e10, true)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	// Scale both a and b by their respective scaling factors.
	scale(a, 1/f[0])
	scale(b, 1/f[1])

	fmt.Println("After normalisation:")
	fmt.Printf("All:    mean(a) = %.1f, mean(b) = %.1f\n", mean(a), mean(b))
	fmt.Printf("Common: mean(a) = %.1f, mean(b) = %.1f\n", mean(a[1e2:]), mean(b[1e2:]))
	fmt.Printf("Up:     mean(a) = %.1f, mean(b) = %.1f\n", mean(a[:1e2]), mean(b[:1e2]))

	// Output:
	// Before normalisation:
	// All:    mean(a) = 9.5, mean(b) = 9.4
	// Common: mean(a) = 9.5, mean(b) = 8.6
	// Up:     mean(a) = 9.4, mean(b) = 91.0
	//
	// After normalisation:
	// All:    mean(a) = 9.1, mean(b) = 9.9
	// Common: mean(a) = 9.1, mean(b) = 9.0
	// Up:     mean(a) = 8.9, mean(b) = 95.4
}

var (
	testData = [][]float64{
		random(1e5, 10, 3), random(1e5, 10, 3), random(1e5, 10, 3), random(1e5, 10, 3), random(1e5, 10, 3),
		random(1e5, 10, 3), random(1e5, 10, 3), random(1e5, 10, 3), random(1e5, 10, 3), random(1e5, 10, 3),
	}
	f []float64
)

func BenchmarkTMMWeighted(b *testing.B) {
	for i := 0; i < b.N; i++ {
		f, _ = norm.TMM(testData, -1, 0.3, 0.05, -1e10, true)
	}
}

func BenchmarkTMMUnweighted(b *testing.B) {
	for i := 0; i < b.N; i++ {
		f, _ = norm.TMM(testData, -1, 0.3, 0.05, -1e10, false)
	}
}

func BenchmarkQuantile(b *testing.B) {
	for i := 0; i < b.N; i++ {
		f, _ = norm.Quantile(testData, 0.75)
	}
}

func BenchmarkRelativeLog(b *testing.B) {
	for i := 0; i < b.N; i++ {
		f, _ = norm.RelativeLog(testData)
	}
}
