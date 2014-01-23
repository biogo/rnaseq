// Copyright ©2013 The bíogo Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package norm provides expression data normalisation functions.
//
// Each of the normalisation factor functions take a slice of float64 data vectors of equal length
// and returns a slice of float64 values that can be used to normalise the vectors by division.
package norm

import (
	"errors"
	"math"
	"sort"
)

// ones returns a slice of float64 n long, and populated with unit values.
func ones(n int) []float64 {
	f := make([]float64, n)
	for i := range f {
		f[i] = 1
	}
	return f
}

// prepare assesses data vectors for correct shape and fast path cases. If no
// fast path is available and the data are correctly formed, the result contains
// the sum of each vector in data.
func prepare(data [][]float64) (done bool, result []float64, err error) {
	rows := len(data[0])
	if rows == 0 {
		return true, ones(len(data)), nil
	}

	cols := len(data)
	if cols == 1 {
		return true, []float64{1}, nil
	}

	size := make([]float64, len(data))
	for i, col := range data {
		if len(col) != rows {
			return true, nil, errors.New("norm: mismatched data vector lengths")
		}
		for _, v := range col {
			size[i] += v
		}
	}

	return false, size, nil
}

// expMeanLogScaled returns the float64 slice scaled by the exp mean of the logged values.
func expMeanLogScaled(f []float64) []float64 {
	var expMeanLog float64
	for _, v := range f {
		expMeanLog += math.Log(v)
	}
	expMeanLog = math.Exp(expMeanLog / float64(len(f)))

	for i, v := range f {
		f[i] = v / expMeanLog
	}

	return f
}

// quantileR7 returns the pth quantile of v according the R-7 method.
// http://en.wikipedia.org/wiki/Quantile#Estimating_the_quantiles_of_a_population
func quantileR7(v []float64, p float64) float64 {
	sort.Float64s(v)
	if p == 1 {
		return v[len(v)-1]
	}
	h := float64(len(v)-1) * p
	i := int(h)
	return v[i] + (h-math.Floor(h))*(v[i+1]-v[i])
}

// quantiles returns the p-quantiles of the size-normalised data vectors.
func quantiles(data [][]float64, skip []bool, size []float64, p float64) []float64 {
	y := make([]float64, 0, len(data[0]))
	q := make([]float64, len(data))
	for i, col := range data {
		sum := size[i]
		for j, v := range col {
			if skip != nil && skip[j] {
				continue
			}
			y = append(y, v/sum)
		}
		q[i] = quantileR7(y, p)
		y = y[:0]
	}
	return q
}

// refIndex returns the preferred reference index. If ref is a valid index into data
// ref is returned unchanged, otherwise the index of the vector with the 75-percentile
// closest to the mean 75-percentile is returned.
func refIndex(data [][]float64, ref int, size []float64) int {
	if 0 <= ref && ref < len(data) {
		return ref
	}

	q75 := quantiles(data, nil, size, 0.75)

	var meanQ75 float64
	for _, v := range q75 {
		meanQ75 += v
	}
	meanQ75 /= float64(len(q75))

	ref = 0
	min := math.Abs(q75[0] - meanQ75)
	for i, v := range q75[1:] {
		if v := math.Abs(v - meanQ75); v < min {
			min = v
			ref = i + 1
		}
	}

	return ref
}

// ranker is a helper type for the rank function.
type ranker struct {
	f []float64 // Data to be ranked.
	r []int     // A list of indexes into f that reflects rank order after sorting.
}

// ranker satisfies the sort.Interface without mutating the reference slice, f.
func (r ranker) Len() int           { return len(r.f) }
func (r ranker) Less(i, j int) bool { return r.f[r.r[i]] < r.f[r.r[j]] }
func (r ranker) Swap(i, j int)      { r.r[i], r.r[j] = r.r[j], r.r[i] }

// rank returns the sample ranks of the values in a vector. Ties (i.e.,
// equal values) are handled by ranking them as the mean rank of coequals.
func (r *ranker) rank(f []float64) []float64 {
	if len(f) == 0 {
		return nil
	}

	r.f = f
	if len(r.r) < len(f) {
		r.r = make([]int, len(f))
	} else {
		r.r = r.r[:len(f)]
	}

	for i := range r.r {
		r.r[i] = i
	}
	sort.Sort(r)
	rl := make([]float64, len(f))
	for i, j := range r.r {
		rl[j] = float64(i)
	}

	var (
		prev = r.f[r.r[0]]

		first int
		same  bool
	)
	for i, j := range r.r[1:] {
		if r.f[j] == prev {
			if !same {
				first = i
			}
			same = true
		} else if same {
			v := (rl[r.r[i]] + rl[r.r[first]]) / 2
			for k := first; k <= i; k++ {
				rl[r.r[k]] = v
			}
			same = false
		}
		prev = r.f[j]
	}

	return rl
}

// tmmFactors returns the relative weighting of each vector compared to a specified
// reference vector according to the TMM normalisation strategy.
func tmmFactors(data [][]float64, refIdx int, size []float64, ratio, sum, minA float64, weight bool) []float64 {
	f := make([]float64, len(data))
	ref := data[refIdx]
	sizeRef := size[refIdx]
	invSizeRef := 1 / sizeRef
	for k, alt := range data {
		sizeAlt := size[k]
		invSizeAlt := 1 / sizeAlt

		// Look for identity between the two vectors to take possible fast path.
		eq := true
		for i, v := range alt {
			if ref[i] != v {
				eq = false
				break
			}
		}
		if eq {
			f[k] = 1
			continue
		}

		var (
			logRat = make([]float64, 0, len(alt))
			logInt = make([]float64, 0, len(alt))
			asmVar []float64
		)
		if weight {
			asmVar = make([]float64, 0, len(alt))
		}
		for i := range alt {
			// Calculate the gene-wise M_g and A_g.
			lR := math.Log2((alt[i] * invSizeAlt) / (ref[i] * invSizeRef))
			aI := math.Log2(alt[i]*invSizeAlt*ref[i]*invSizeRef) / 2

			// Reject all disallowed data points here.
			if aI < minA || math.IsInf(lR, 0) || math.IsNaN(lR) || math.IsInf(aI, 0) || math.IsNaN(aI) {
				continue
			}

			logRat = append(logRat, lR)
			logInt = append(logInt, aI)

			// Calculate asymptotic variance if weighting is requested.
			if weight {
				asmVar = append(asmVar, (sizeAlt-alt[i])*invSizeAlt/alt[i]+(sizeRef-ref[i])*invSizeRef/ref[i])
			}
		}

		// Determine the starts of tails that we trim.
		n := float64(len(logRat))
		minRat := math.Floor(n * ratio)
		maxRat := n - minRat - 1
		minSum := math.Floor(n * sum)
		maxSum := n - minSum - 1

		var r ranker
		rLogRat := r.rank(logRat)
		rLogInt := r.rank(logInt)

		var num, den float64
		for i := range logRat {
			// Trim by log fold-change and absolute intensity.
			if rLogRat[i] < minRat || rLogRat[i] > maxRat || rLogInt[i] < minSum || rLogInt[i] > maxSum {
				continue
			}

			// Weight by asymptotic variance if requested.
			if weight {
				num += (logRat[i] / asmVar[i])
				den += 1 / asmVar[i]
			} else {
				num += logRat[i]
				den++
			}
		}

		f[k] = math.Pow(2, num/den)
	}

	return f
}

// TMM returns a slice of factors that normalise the data vectors according the
// TMM normalisation strategy. The value of ref specifies which vector in data
// is to be used a the reference vector. If ref is not a valid index into data,
// the reference is the vector with the 75-percentile closest to the mean of
// the vectors' 75-percentiles.
//
// "A scaling normalization method for differential expression analysis of RNA-seq data",
// Mark Robinson and Alicia Oshlack, http://genomebiology.com/2010/11/3/r25.
func TMM(data [][]float64, ref int, ratio, sum, minA float64, weight bool) ([]float64, error) {
	if len(data) == 0 {
		return nil, nil
	}

	done, result, err := prepare(data)
	if done {
		return result, err
	}
	size := result

	f := tmmFactors(data, refIndex(data, ref, size), size, ratio, sum, minA, weight)

	return expMeanLogScaled(f), nil
}

// skipAllZeros returns a slice of bool indicating which rows to skip due to
// being all zero.
func skipAllZeros(data [][]float64) []bool {
	skip := make([]bool, len(data[0]))
	for i := range data[0] {
		skip[i] = true
		for _, col := range data {
			if col[i] != 0 {
				skip[i] = false
				break
			}
		}
	}
	return skip
}

// Quantile returns a slice of factors that normalise the data vectors according to a
// generalisation of the upper quartile normalisation strategy.
//
// "Evaluation of statistical methods for normalization and differential expression in mRNA-Seq
// experiments", James Bullard et al., http://www.biomedcentral.com/1471-2105/11/94.
func Quantile(data [][]float64, p float64) ([]float64, error) {
	if len(data) == 0 {
		return nil, nil
	}

	done, result, err := prepare(data)
	if done {
		return result, err
	}
	size := result

	// S4 in http://www.biomedcentral.com/content/supplementary/1471-2105-11-94-s2.pdf.
	f := quantiles(data, skipAllZeros(data), size, p)

	return expMeanLogScaled(f), nil
}

// rleFactors returns the medians of ratios of observed values. eq. 5 in
// http://genomebiology.com/2010/11/10/r106.
func rleFactors(data [][]float64, skip []bool, size []float64) []float64 {
	xm := make([]float64, len(data[0]))
	for i := range data[0] {
		if skip[i] {
			continue
		}
		var v float64
		// Calculate the mean ratios in log space.
		for j := range data {
			v += math.Log(data[j][i])
		}
		v /= float64(len(data))
		xm[i] = math.Exp(v)
	}
	t := make([]float64, 0, len(xm))
	f := make([]float64, len(data))
	for j, col := range data {
		for i, v := range col {
			if skip[i] || xm[i] == 0 {
				continue
			}
			// Append k_ij / (∏v=1..m(k_iv)^(1/m))
			t = append(t, v/xm[i])
		}
		f[j] = quantileR7(t, 0.5) / size[j] // Keep normalised median.
		t = t[:0]
	}
	return f
}

// RelativeLog returns a slice of factors that normalise the data vectors according the
// relative log expression normalisation strategy.
//
// "Differential expression analysis for sequence count data", Simon Anders and Wolfgang Huber,
// http://genomebiology.com/2010/11/10/r106.
func RelativeLog(data [][]float64) ([]float64, error) {
	if len(data) == 0 {
		return nil, nil
	}

	done, result, err := prepare(data)
	if done {
		return result, err
	}
	size := result

	f := rleFactors(data, skipAllZeros(data), size)

	return expMeanLogScaled(f), nil
}
