;;;; buddhabrot-512.lisp — High-Performance AVX-512 Buddhabrot Generator
;;;;
;;;; Clean-room implementation based on the Buddhabrot technique (Melinda Green, 1993).
;;;;
;;;; Copyright (c) 2026 Arthur Miller <arthur.miller@live.com>
;;;;
;;;; Redistribution and use in source and binary forms, with or without
;;;; modification, are permitted provided that the following conditions are met:
;;;;
;;;; 1. Redistributions of source code must retain the above copyright notice,
;;;;    this list of conditions and the following disclaimer.
;;;; 2. Redistributions in binary form must reproduce the above copyright notice,
;;;;    this list of conditions and the following disclaimer in the documentation
;;;;    and/or other materials provided with the distribution.
;;;;
;;;; THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
;;;; AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
;;;; IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
;;;; ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
;;;; LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
;;;; CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
;;;; SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
;;;; INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
;;;; CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
;;;; ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
;;;; POSSIBILITY OF SUCH DAMAGE.

(in-package :cl-user)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (require :sb-simd))

(defpackage #:buddha
  (:use #:cl #:sb-simd-avx512bw)
  (:export #:render
           #:export-exr
           #:denoise-buffer
           #:draw
           #:main
           #:main-denoise
           #:run-thread
           #:width
           #:height
           #:*color-array*))

(in-package :buddha)

(setf *read-default-float-format* 'double-float)
(declaim (optimize (speed 3) (safety 0) (debug 0)))
(sb-int:set-floating-point-modes :traps nil)

;;; ============================================================================
;;; Image & Coordinate Constants
;;; ============================================================================

(defconstant +default-width+ 12944)
(defconstant +default-height+ 8000)

;; Double-precision coordinate space
(defconstant +image-x+ 8.045d0)
(defconstant +image-y+ 5.0d0)

(defconstant +start-image-x+ -4.045d0)
(defconstant +start-image-y+ 2.5d0)
(defconstant +end-image-x+ 4.0d0)
(defconstant +end-image-y+ -2.5d0)

(defconstant +min-x+ -2.045d0)
(defconstant +max-x+ 2.0d0)
(defconstant +min-y+ 0.0d0)
(defconstant +max-y+ 1.25d0)

(defconstant +escape-radius-sq+ 4.0d0)
(defconstant +max-bailout+ 100000)
(defconstant +min-bailout+ 20)
(defconstant +period-2-bulb-radius-sq+ 0.0625d0) ; (1/4)^2 = 1/16

;; Single-precision coordinate space (for high-throughput 16-wide AVX-512 engine)
(defconstant +image-x-f32+ 8.045f0)
(defconstant +image-y-f32+ 5.0f0)

(defconstant +start-image-x-f32+ -4.045f0)
(defconstant +start-image-y-f32+ 2.5f0)
(defconstant +end-image-x-f32+ 4.0f0)
(defconstant +end-image-y-f32+ -2.5f0)

(defconstant +min-x-f32+ -2.045f0)
(defconstant +max-x-f32+ 2.0f0)
(defconstant +min-y-f32+ 0.0f0)
(defconstant +max-y-f32+ 1.25f0)

(defconstant +escape-radius-sq-f32+ 4.0f0)
(defconstant +period-2-bulb-radius-sq-f32+ 0.0625f0)

;; Periodicity detection constants (catches trapped attracting basins before bailout)
(defconstant +cycle-detection-interval+ 2520) ; lcm(1..10) catches periods 1..10
(defconstant +cycle-eps-sq-f32+ 1.0f-7)
(defconstant +cycle-eps-sq-f64+ 1.0d-14)

(sb-ext:defglobal *default-output-filename* "buddhabrot.exr")

;; Backwards compatibility aliases
(defparameter width +default-width+)
(defparameter height +default-height+)

;;; The 103.55 Megapixel image buffer uses 32-bit unsigned integers per channel (R, G, B).
;;; Even with 80,000 escaped orbits, the maximum peak hits observed in any single pixel
;;; is ~1,500. Overflowing (unsigned-byte 32) requires ~4.29 billion hits in a single pixel
;;; (equivalent to ~200,000x more sampling passes). Switching from 64-bit to 32-bit halves
;;; physical RAM footprint from 2.48 GB to 1.24 GB and halves memory-bus traffic across
;;; buffer clearing, AVX-512 retrace hit accumulation, and channel maximum scanning.
(deftype image-buffer () '(simple-array (unsigned-byte 32) (*)))

(sb-ext:defglobal *color-array*
  (make-array (* +default-width+ +default-height+ 3)
              :element-type '(unsigned-byte 32)
              :initial-element 0))

;;; ============================================================================
;;; Fast Unboxed PRNG: Xoshiro256+
;;; ============================================================================

(declaim (inline rotl))
(defun rotl (x k)
  (declare (type (unsigned-byte 64) x)
           (type (integer 0 64) k))
  (logand #xFFFFFFFFFFFFFFFF
          (logior (ash x k)
                  (ash x (- k 64)))))

(defstruct (xoshiro (:constructor %make-xoshiro))
  (s0 0 :type (unsigned-byte 64))
  (s1 0 :type (unsigned-byte 64))
  (s2 0 :type (unsigned-byte 64))
  (s3 0 :type (unsigned-byte 64)))

(defun make-fast-rng (seed-int)
  "Initialize a Xoshiro256+ generator using SplitMix64 on a seed integer."
  (declare (type (unsigned-byte 64) seed-int))
  (let ((cur seed-int))
    (flet ((splitmix64 ()
             (setf cur (logand #xFFFFFFFFFFFFFFFF (+ cur #x9E3779B97F4A7C15)))
             (let ((z cur))
               (setf z (logand #xFFFFFFFFFFFFFFFF
                               (* (logxor z (ash z -30)) #xBF58476D1CE4E5B9)))
               (setf z (logand #xFFFFFFFFFFFFFFFF
                               (* (logxor z (ash z -27)) #x94D049BB133111EB)))
               (logxor z (ash z -31)))))
      (%make-xoshiro :s0 (splitmix64)
                     :s1 (splitmix64)
                     :s2 (splitmix64)
                     :s3 (splitmix64)))))

(declaim (inline xoshiro-next-u64))
(defun xoshiro-next-u64 (rng)
  (declare (type xoshiro rng))
  (let* ((s0 (xoshiro-s0 rng))
         (s1 (xoshiro-s1 rng))
         (s2 (xoshiro-s2 rng))
         (s3 (xoshiro-s3 rng))
         (result (logand #xFFFFFFFFFFFFFFFF (+ s0 s3)))
         (tt (logand #xFFFFFFFFFFFFFFFF (ash s1 17))))
    (declare (type (unsigned-byte 64) s0 s1 s2 s3 result tt))
    (setf s2 (logxor s2 s0)
          s3 (logxor s3 s1)
          s1 (logxor s1 s2)
          s0 (logxor s0 s3)
          s2 (logxor s2 tt)
          s3 (rotl s3 45))
    (setf (xoshiro-s0 rng) s0
          (xoshiro-s1 rng) s1
          (xoshiro-s2 rng) s2
          (xoshiro-s3 rng) s3)
    result))

(declaim (inline rng-double))
(defun rng-double (rng min max)
  "Generate a double-float in [min, max) with 53 bits of precision."
  (declare (type xoshiro rng)
           (type double-float min max))
  (let* ((bits (ash (xoshiro-next-u64 rng) -11))
         (norm (* (float bits 1d0) 1.1102230246251565d-16))) ; 2^-53
    (+ min (* norm (- max min)))))

(declaim (inline rng-single))
(defun rng-single (rng min max)
  "Generate a single-float in [min, max) with 24 bits of precision directly from u64."
  (declare (type xoshiro rng)
           (type single-float min max))
  (let* ((bits (ldb (byte 24 40) (xoshiro-next-u64 rng)))
         (norm (* (float bits 1.0f0) #.(/ 1.0f0 16777216.0f0))))
    (+ min (* norm (- max min)))))

;;; ============================================================================
;;; Cardioid, Period-2 Bulb, and Candidate Sampling
;;; ============================================================================

(declaim (inline inside-cardioid-or-bulb-p))
(defun inside-cardioid-or-bulb-p (cr ci)
  "Analytically detect if candidate c is inside the main cardioid or period-2 bulb (f64)."
  (declare (type double-float cr ci))
  (let* ((cr-q (- cr 0.25d0))
         (ci2 (* ci ci))
         (q (+ ci2 (* cr-q cr-q)))
         (cr-p (+ cr 1.0d0))
         (s (+ ci2 (* cr-p cr-p))))
    (or (<= (* q (+ q cr-q)) (* 0.25d0 ci2)) ; Main cardioid
        (<= s +period-2-bulb-radius-sq+))))   ; Period-2 bulb

(declaim (inline generate-candidate))
(defun generate-candidate (rng)
  "Generate a random candidate in [min, max] outside the cardioid and period-2 bulb (f64)."
  (declare (type xoshiro rng))
  (loop
    (let ((cr (rng-double rng +min-x+ +max-x+))
          (ci (rng-double rng +min-y+ +max-y+)))
      (declare (type double-float cr ci))
      (when (and (< (+ (* cr cr) (* ci ci)) +escape-radius-sq+)
                 (not (inside-cardioid-or-bulb-p cr ci)))
        (return (values cr ci))))))

(declaim (inline inside-cardioid-or-bulb-f32-p))
(defun inside-cardioid-or-bulb-f32-p (cr ci)
  "Analytically detect if candidate c is inside the main cardioid or period-2 bulb (f32)."
  (declare (type single-float cr ci))
  (let* ((cr-q (- cr 0.25f0))
         (ci2 (* ci ci))
         (q (+ ci2 (* cr-q cr-q)))
         (cr-p (+ cr 1.0f0))
         (s (+ ci2 (* cr-p cr-p))))
    (declare (type single-float cr-q ci2 q cr-p s))
    (or (<= (* q (+ q cr-q)) (* 0.25f0 ci2))
        (<= s +period-2-bulb-radius-sq-f32+))))

(declaim (inline generate-candidate-f32))
(defun generate-candidate-f32 (rng)
  "Generate a random candidate in [min, max] outside the cardioid and period-2 bulb (f32)."
  (declare (type xoshiro rng))
  (loop
    (let ((cr (rng-single rng +min-x-f32+ +max-x-f32+))
          (ci (rng-single rng +min-y-f32+ +max-y-f32+)))
      (declare (type single-float cr ci))
      (when (and (< (+ (* cr cr) (* ci ci)) +escape-radius-sq-f32+)
                 (not (inside-cardioid-or-bulb-f32-p cr ci)))
        (return (values cr ci))))))

;;; ============================================================================
;;; Spectral Channel Classification (Nebula Technique)
;;; ============================================================================

(declaim (inline orbit-channel))
(defun orbit-channel (steps)
  "Partition escaped orbits into Red, Green, and Blue frequency bands."
  (declare (type (unsigned-byte 32) steps))
  (cond
    ((>= steps 1000) 0)   ; Red: Deep, long-lived orbits
    ((>= steps 100)  1)   ; Green: Medium-depth orbits
    (t               2))) ; Blue: Short, shallow orbits

;;; ============================================================================
;;; Symmetric Orbit Projection (Patches 1, 2, 3 Optimized)
;;; ============================================================================

(declaim (inline trace-symmetric-orbit-f32))
(defun trace-symmetric-orbit-f32 (cr ci steps array w h px-mul x-off py-mul y-off)
  "Retrace an escaped orbit and project (x, y) and (x, -y) using folded offsets, mirror, and plain incf."
  (declare (type single-float cr ci px-mul x-off py-mul y-off)
           (type (unsigned-byte 32) steps)
           (type image-buffer array)
           (type fixnum w h))
  (let ((channel (orbit-channel steps))
        (zr cr)
        (zi ci)
        (max-x (1- w))
        (max-y (1- h))
        (h-1   (1- h)))
    (declare (type single-float zr zi)
             (type fixnum channel max-x max-y h-1))
    (dotimes (k steps)
      (let ((mag2 (+ (* zr zr) (* zi zi))))
        (when (>= mag2 +escape-radius-sq-f32+)
          (return)))
      (let* ((px (floor (+ (* zr px-mul) x-off)))
             (py (floor (+ (* zi py-mul) y-off)))
             (py-conj (- h-1 py)))
        (declare (type fixnum px py py-conj))
        (when (and (<= 0 px max-x) (<= 0 py max-y))
          (incf (aref array (+ channel (the fixnum (* (+ (the fixnum (* py w)) px) 3))))))
        (when (and (<= 0 px max-x) (<= 0 py-conj max-y) (/= py py-conj))
          (incf (aref array (+ channel (the fixnum (* (+ (the fixnum (* py-conj w)) px) 3)))))))
      (let ((next-zr (+ (- (* zr zr) (* zi zi)) cr))
            (next-zi (+ (* 2.0f0 zr zi) ci)))
        (setf zr next-zr
              zi next-zi)))))

(declaim (inline trace-symmetric-orbit-f64))
(defun trace-symmetric-orbit-f64 (cr ci steps array w h px-mul x-off py-mul y-off)
  "Retrace double-precision orbit using folded offsets, mirror, and plain incf."
  (declare (type double-float cr ci px-mul x-off py-mul y-off)
           (type (unsigned-byte 32) steps)
           (type image-buffer array)
           (type fixnum w h))
  (let ((channel (orbit-channel steps))
        (zr cr)
        (zi ci)
        (max-x (1- w))
        (max-y (1- h))
        (h-1   (1- h)))
    (declare (type double-float zr zi)
             (type fixnum channel max-x max-y h-1))
    (dotimes (k steps)
      (let ((mag2 (+ (* zr zr) (* zi zi))))
        (when (>= mag2 +escape-radius-sq+)
          (return)))
      (let* ((px (floor (+ (* zr px-mul) x-off)))
             (py (floor (+ (* zi py-mul) y-off)))
             (py-conj (- h-1 py)))
        (declare (type fixnum px py py-conj))
        (when (and (<= 0 px max-x) (<= 0 py max-y))
          (incf (aref array (+ channel (the fixnum (* (+ (the fixnum (* py w)) px) 3))))))
        (when (and (<= 0 px max-x) (<= 0 py-conj max-y) (/= py py-conj))
          (incf (aref array (+ channel (the fixnum (* (+ (the fixnum (* py-conj w)) px) 3)))))))
      (let ((next-zr (+ (- (* zr zr) (* zi zi)) cr))
            (next-zi (+ (* 2.0d0 zr zi) ci)))
        (setf zr next-zr
              zi next-zi)))))

(declaim (inline trace-symmetric-orbit))
(defun trace-symmetric-orbit (cr ci steps array w h scale-x scale-y)
  "Backward-compatible trace-symmetric-orbit wrapper."
  (declare (type double-float cr ci scale-x scale-y)
           (type (unsigned-byte 32) steps)
           (type image-buffer array)
           (type fixnum w h))
  (let ((px-mul scale-x)
        (py-mul scale-y)
        (x-off (* (- +start-image-x+) scale-x))
        (y-off (* (- +end-image-y+) scale-y)))
    (trace-symmetric-orbit-f64 cr ci steps array w h px-mul x-off py-mul y-off)))

;;; ============================================================================
;;; AVX-512 Vector Engine (Continuous Lane Refill)
;;; ============================================================================

(defun simd-worker-f32 (target-hits rng array w h &optional (cycle-detection t))
  "Compute Mandelbrot orbits 16-wide using AVX-512 single-precision registers with cycle detection."
  (declare (type fixnum target-hits w h)
           (type xoshiro rng)
           (type image-buffer array)
           (type boolean cycle-detection)
           (optimize (speed 3) (safety 0) (debug 0)))
  (let* ((cr-arr (make-array 16 :element-type 'single-float))
         (ci-arr (make-array 16 :element-type 'single-float))
         (zr-arr (make-array 16 :element-type 'single-float))
         (zi-arr (make-array 16 :element-type 'single-float))
         (start-steps (make-array 16 :element-type '(unsigned-byte 32) :initial-element 0))
         (snap-zr (make-array 16 :element-type 'single-float :initial-element 0.0f0))
         (snap-zi (make-array 16 :element-type 'single-float :initial-element 0.0f0))
         (snap-valid (make-array 16 :element-type 'bit :initial-element 0))
         (four (sb-simd-avx512f:f32.16-broadcast +escape-radius-sq-f32+))
         (scale-x (float (/ (float w 1d0) +image-x+) 1.0f0))
         (scale-y (float (/ (float h 1d0) +image-y+) 1.0f0))
         (px-mul scale-x)
         (py-mul scale-y)
         (x-off (* (- +start-image-x-f32+) scale-x))
         (y-off (* (- +end-image-y-f32+) scale-y))
         (hits 0)
         (global-step 0)
         (steps-since-check 0))
    (declare (type (simple-array single-float (16)) cr-arr ci-arr zr-arr zi-arr snap-zr snap-zi)
             (type (simple-array (unsigned-byte 32) (16)) start-steps)
             (type (simple-array bit (16)) snap-valid)
             (type single-float px-mul py-mul x-off y-off)
             (type fixnum hits steps-since-check)
             (type (unsigned-byte 32) global-step)
             (type sb-simd-avx512f:f32.16 four))
    ;; Populate initial 16 lanes
    (dotimes (i 16)
      (multiple-value-bind (cr ci) (generate-candidate-f32 rng)
        (setf (aref cr-arr i) cr
              (aref ci-arr i) ci
              (aref zr-arr i) cr
              (aref zi-arr i) ci
              (aref start-steps i) 0
              (aref snap-valid i) 0)))
    (let ((cr-vec (sb-simd-avx512f:f32.16-aref cr-arr 0))
          (ci-vec (sb-simd-avx512f:f32.16-aref ci-arr 0))
          (zr-vec (sb-simd-avx512f:f32.16-aref zr-arr 0))
          (zi-vec (sb-simd-avx512f:f32.16-aref zi-arr 0)))
      (declare (type sb-simd-avx512f:f32.16 cr-vec ci-vec zr-vec zi-vec))
      (loop while (< hits target-hits) do
        (incf global-step)
        ;; 512-bit vector Mandelbrot step (16 lanes in parallel)
        (let* ((zr2 (sb-simd-avx512f:f32.16* zr-vec zr-vec))
               (zi2 (sb-simd-avx512f:f32.16* zi-vec zi-vec))
               (zri (sb-simd-avx512f:f32.16* zr-vec zi-vec))
               (mag2 (sb-simd-avx512f:f32.16+ zr2 zi2))
               (escaped (sb-simd-avx512f:f32.16-mask>= mag2 four))
               (mask (sb-kernel:%simd-pack-512-mask-value escaped))
               (check-bailout (>= (incf steps-since-check) +cycle-detection-interval+)))
          (declare (type (unsigned-byte 16) mask))
          (if (and (zerop mask) (not check-bailout))
              ;; Fast Path: 100% in ZMM registers, zero memory operations
              (setf zr-vec (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16- zr2 zi2) cr-vec)
                    zi-vec (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16+ zri zri) ci-vec))
              ;; Slow Path: One or more lanes escaped, timed out, or reached period check window
              (progn
                (when check-bailout (setf steps-since-check 0))
                (setf (sb-simd-avx512f:f32.16-aref zr-arr 0) (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16- zr2 zi2) cr-vec)
                      (sb-simd-avx512f:f32.16-aref zi-arr 0) (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16+ zri zri) ci-vec))
                (dotimes (i 16)
                  (let ((steps (- global-step (aref start-steps i))))
                    (declare (type (unsigned-byte 32) steps))
                    (cond
                      ((logbitp i mask)
                       ;; Lane escaped!
                       (when (<= +min-bailout+ steps +max-bailout+)
                         (trace-symmetric-orbit-f32 (aref cr-arr i) (aref ci-arr i) steps
                                                    array w h px-mul x-off py-mul y-off)
                         (incf hits))
                       ;; Refill escaped lane with a new candidate
                       (multiple-value-bind (cr ci) (generate-candidate-f32 rng)
                         (setf (aref cr-arr i) cr
                               (aref ci-arr i) ci
                               (aref zr-arr i) cr
                               (aref zi-arr i) ci
                               (aref start-steps i) global-step
                               (aref snap-valid i) 0)))
                      ;; Cycle detection on alive lanes
                      ((and cycle-detection
                            check-bailout
                            (= (aref snap-valid i) 1)
                            (let ((dzr (- (aref zr-arr i) (aref snap-zr i)))
                                  (dzi (- (aref zi-arr i) (aref snap-zi i))))
                              (< (+ (* dzr dzr) (* dzi dzi)) +cycle-eps-sq-f32+)))
                       ;; Trapped in periodic cycle! Refill lane immediately
                       (multiple-value-bind (cr ci) (generate-candidate-f32 rng)
                         (setf (aref cr-arr i) cr
                               (aref ci-arr i) ci
                               (aref zr-arr i) cr
                               (aref zi-arr i) ci
                               (aref start-steps i) global-step
                               (aref snap-valid i) 0)))
                      ((>= steps +max-bailout+)
                       ;; Lane timed out
                       (multiple-value-bind (cr ci) (generate-candidate-f32 rng)
                         (setf (aref cr-arr i) cr
                               (aref ci-arr i) ci
                               (aref zr-arr i) cr
                               (aref zi-arr i) ci
                               (aref start-steps i) global-step
                               (aref snap-valid i) 0)))
                      (check-bailout
                       (when cycle-detection
                         (setf (aref snap-zr i) (aref zr-arr i)
                               (aref snap-zi i) (aref zi-arr i)
                               (aref snap-valid i) 1))))))
                ;; Reload ZMM vector registers
                (setf cr-vec (sb-simd-avx512f:f32.16-aref cr-arr 0)
                      ci-vec (sb-simd-avx512f:f32.16-aref ci-arr 0)
                      zr-vec (sb-simd-avx512f:f32.16-aref zr-arr 0)
                      zi-vec (sb-simd-avx512f:f32.16-aref zi-arr 0)))))))))

(defun simd-worker-f64 (target-hits rng array w h &optional (cycle-detection t))
  "Compute Mandelbrot orbits 8-wide using AVX-512 double-precision registers with cycle detection."
  (declare (type fixnum target-hits w h)
           (type xoshiro rng)
           (type image-buffer array)
           (type boolean cycle-detection)
           (optimize (speed 3) (safety 0) (debug 0)))
  (let* ((cr-arr (make-array 8 :element-type 'double-float))
         (ci-arr (make-array 8 :element-type 'double-float))
         (zr-arr (make-array 8 :element-type 'double-float))
         (zi-arr (make-array 8 :element-type 'double-float))
         (start-steps (make-array 8 :element-type '(unsigned-byte 32) :initial-element 0))
         (snap-zr (make-array 8 :element-type 'double-float :initial-element 0.0d0))
         (snap-zi (make-array 8 :element-type 'double-float :initial-element 0.0d0))
         (snap-valid (make-array 8 :element-type 'bit :initial-element 0))
         (four (f64.8-broadcast +escape-radius-sq+))
         (scale-x (/ (float w 1d0) +image-x+))
         (scale-y (/ (float h 1d0) +image-y+))
         (px-mul scale-x)
         (py-mul scale-y)
         (x-off (* (- +start-image-x+) scale-x))
         (y-off (* (- +end-image-y+) scale-y))
         (hits 0)
         (global-step 0)
         (steps-since-check 0))
    (declare (type (simple-array double-float (8)) cr-arr ci-arr zr-arr zi-arr snap-zr snap-zi)
             (type (simple-array (unsigned-byte 32) (8)) start-steps)
             (type (simple-array bit (8)) snap-valid)
             (type double-float px-mul py-mul x-off y-off)
             (type fixnum hits steps-since-check)
             (type (unsigned-byte 32) global-step)
             (type f64.8 four))
    ;; Populate initial 8 lanes
    (dotimes (i 8)
      (multiple-value-bind (cr ci) (generate-candidate rng)
        (setf (aref cr-arr i) cr
              (aref ci-arr i) ci
              (aref zr-arr i) cr
              (aref zi-arr i) ci
              (aref start-steps i) 0
              (aref snap-valid i) 0)))
    (let ((cr-vec (f64.8-aref cr-arr 0))
          (ci-vec (f64.8-aref ci-arr 0))
          (zr-vec (f64.8-aref zr-arr 0))
          (zi-vec (f64.8-aref zi-arr 0)))
      (declare (type f64.8 cr-vec ci-vec zr-vec zi-vec))
      (loop while (< hits target-hits) do
        (incf global-step)
        ;; 512-bit vector Mandelbrot step (8 lanes in parallel)
        (let* ((zr2 (f64.8* zr-vec zr-vec))
               (zi2 (f64.8* zi-vec zi-vec))
               (zri (f64.8* zr-vec zi-vec))
               (mag2 (f64.8+ zr2 zi2))
               (escaped (f64.8-mask>= mag2 four))
               (mask (sb-kernel:%simd-pack-512-mask-value escaped))
               (check-bailout (>= (incf steps-since-check) +cycle-detection-interval+)))
          (declare (type (unsigned-byte 8) mask))
          (if (and (zerop mask) (not check-bailout))
              ;; Fast Path: 100% in ZMM registers, zero memory operations
              (setf zr-vec (f64.8+ (f64.8- zr2 zi2) cr-vec)
                    zi-vec (f64.8+ (f64.8+ zri zri) ci-vec))
              ;; Slow Path
              (progn
                (when check-bailout (setf steps-since-check 0))
                (setf (f64.8-aref zr-arr 0) (f64.8+ (f64.8- zr2 zi2) cr-vec)
                      (f64.8-aref zi-arr 0) (f64.8+ (f64.8+ zri zri) ci-vec))
                (dotimes (i 8)
                  (let ((steps (- global-step (aref start-steps i))))
                    (declare (type (unsigned-byte 32) steps))
                    (cond
                      ((logbitp i mask)
                       ;; Lane escaped!
                       (when (<= +min-bailout+ steps +max-bailout+)
                         (trace-symmetric-orbit-f64 (aref cr-arr i) (aref ci-arr i) steps
                                                    array w h px-mul x-off py-mul y-off)
                         (incf hits))
                       (multiple-value-bind (cr ci) (generate-candidate rng)
                         (setf (aref cr-arr i) cr
                               (aref ci-arr i) ci
                               (aref zr-arr i) cr
                               (aref zi-arr i) ci
                               (aref start-steps i) global-step
                               (aref snap-valid i) 0)))
                      ;; Cycle detection on alive lanes
                      ((and cycle-detection
                            check-bailout
                            (= (aref snap-valid i) 1)
                            (let ((dzr (- (aref zr-arr i) (aref snap-zr i)))
                                  (dzi (- (aref zi-arr i) (aref snap-zi i))))
                              (< (+ (* dzr dzr) (* dzi dzi)) +cycle-eps-sq-f64+)))
                       ;; Trapped in periodic cycle! Refill lane immediately
                       (multiple-value-bind (cr ci) (generate-candidate rng)
                         (setf (aref cr-arr i) cr
                               (aref ci-arr i) ci
                               (aref zr-arr i) cr
                               (aref zi-arr i) ci
                               (aref start-steps i) global-step
                               (aref snap-valid i) 0)))
                      ((>= steps +max-bailout+)
                       ;; Lane timed out
                       (multiple-value-bind (cr ci) (generate-candidate rng)
                         (setf (aref cr-arr i) cr
                               (aref ci-arr i) ci
                               (aref zr-arr i) cr
                               (aref zi-arr i) ci
                               (aref start-steps i) global-step
                               (aref snap-valid i) 0)))
                      (check-bailout
                       (when cycle-detection
                         (setf (aref snap-zr i) (aref zr-arr i)
                               (aref snap-zi i) (aref zi-arr i)
                               (aref snap-valid i) 1))))))
                ;; Reload ZMM vector registers
                (setf cr-vec (f64.8-aref cr-arr 0)
                      ci-vec (f64.8-aref ci-arr 0)
                      zr-vec (f64.8-aref zr-arr 0)
                      zi-vec (f64.8-aref zi-arr 0)))))))))

(defun simd-worker (target-hits rng array w h scale-x scale-y)
  "Backward-compatible simd-worker wrapper (defaults to fast f32.16)."
  (declare (ignore scale-x scale-y))
  (simd-worker-f32 target-hits rng array w h nil))

;;; Channel Dynamic Range Analysis
;;; ============================================================================

(defun find-channel-maximums (array length &optional (threads 16))
  "Parallel channel maximum scan across Red, Green, and Blue channels."
  (declare (type image-buffer array)
           (type fixnum length threads)
           (optimize (speed 3) (safety 0) (debug 0)))
  (let* ((pixels (floor length 3))
         (pixels-per-thread (ceiling pixels threads))
         (words-per-thread (* pixels-per-thread 3))
         (results (make-array threads :initial-element nil))
         (worker-list '()))
    (dotimes (t-idx threads)
      (let ((tid t-idx)
            (s (* t-idx words-per-thread))
            (e (min length (* (1+ t-idx) words-per-thread))))
        (when (< s e)
          (push (sb-thread:make-thread
                 (lambda ()
                   (let ((mr 1) (mg 1) (mb 1))
                     (declare (type (unsigned-byte 32) mr mg mb))
                     (do ((i s (+ i 3)))
                         ((>= i e))
                       (let ((vr (aref array i))
                             (vg (aref array (+ i 1)))
                             (vb (aref array (+ i 2))))
                         (declare (type (unsigned-byte 32) vr vg vb))
                         (when (> vr mr) (setf mr vr))
                         (when (> vg mg) (setf mg vg))
                         (when (> vb mb) (setf mb vb))))
                     (setf (aref results tid) (list mr mg mb))))
                 :name (format nil "max-scan-~D" t-idx))
                worker-list))))
    (mapc #'sb-thread:join-thread worker-list)
    (let ((mr 1) (mg 1) (mb 1))
      (declare (type (unsigned-byte 32) mr mg mb))
      (dotimes (i threads)
        (when (aref results i)
          (destructuring-bind (r g b) (aref results i)
            (declare (type (unsigned-byte 32) r g b))
            (when (> r mr) (setf mr r))
            (when (> g mg) (setf mg g))
            (when (> b mb) (setf mb b)))))
      (values mr mg mb))))

(defun parallel-fill (array length &optional (threads 16))
  "Parallel zeroing of the image buffer across worker threads."
  (declare (type image-buffer array)
           (type fixnum length threads)
           (optimize (speed 3) (safety 0) (debug 0)))
  (let* ((chunk (ceiling length threads))
         (workers
           (loop for t-idx below threads collect
             (let ((s (* t-idx chunk))
                   (e (min length (* (1+ t-idx) chunk))))
               (when (< s e)
                 (sb-thread:make-thread
                  (lambda ()
                    (declare (type fixnum s e))
                    (do ((i s (1+ i))) ((>= i e))
                      (setf (aref array i) 0)))
                  :name (format nil "fill-worker-~D" t-idx)))))))
    (mapc #'sb-thread:join-thread (remove nil workers))))

;;; ============================================================================
;;; OpenEXR RLE Compressor (EXR Type 1: Fast Run-Length Encoding)
;;; ============================================================================

(declaim (inline count-run-simd))
(defun count-run-simd (arr pos max-len)
  (declare (type (simple-array (unsigned-byte 8) (*)) arr)
           (type fixnum pos max-len)
           (optimize (speed 3) (safety 0) (debug 0)))
  (let ((val (aref arr pos)))
    (declare (type (unsigned-byte 8) val))
    (if (and (<= (+ pos 64) max-len))
        (let* ((v (u8.64-aref arr pos))
               (b (u8.64-broadcast val))
               (mask (two-arg-u8.64-mask= v b))
               (int-mask (sb-kernel:%simd-pack-512-mask-value mask))
               (diff (logxor int-mask #xFFFFFFFFFFFFFFFF)))
          (declare (type (unsigned-byte 64) int-mask diff))
          (if (zerop diff)
              (if (and (<= (+ pos 128) max-len))
                  (let* ((v2 (u8.64-aref arr (+ pos 64)))
                         (mask2 (two-arg-u8.64-mask= v2 b))
                         (int-mask2 (sb-kernel:%simd-pack-512-mask-value mask2))
                         (diff2 (logxor int-mask2 #xFFFFFFFFFFFFFFFF)))
                    (declare (type (unsigned-byte 64) int-mask2 diff2))
                    (if (zerop diff2)
                        128
                        (+ 64 (1- (integer-length (logand diff2 (- diff2)))))))
                  (let ((run-end (+ pos 64))
                        (limit (min max-len (+ pos 128))))
                    (loop while (and (< run-end limit)
                                     (= (aref arr run-end) val))
                          do (incf run-end))
                    (- run-end pos)))
              (1- (integer-length (logand diff (- diff))))))
        (let ((run-end (1+ pos))
              (limit (min max-len (+ pos 128))))
          (loop while (and (< run-end limit)
                           (= (aref arr run-end) val))
                do (incf run-end))
          (- run-end pos)))))

(declaim (inline find-3run-simd))
(defun find-3run-simd (arr pos max-len)
  (declare (type (simple-array (unsigned-byte 8) (*)) arr)
           (type fixnum pos max-len)
           (optimize (speed 3) (safety 0) (debug 0)))
  (let ((cur pos))
    (declare (type fixnum cur))
    (loop while (and (<= (+ cur 66) max-len)
                     (< (- cur pos) 66))
          do
          (let* ((v0 (u8.64-aref arr cur))
                 (v1 (u8.64-aref arr (+ cur 1)))
                 (v2 (u8.64-aref arr (+ cur 2)))
                 (m1 (sb-kernel:%simd-pack-512-mask-value (two-arg-u8.64-mask= v0 v1)))
                 (m2 (sb-kernel:%simd-pack-512-mask-value (two-arg-u8.64-mask= v1 v2)))
                 (m (logand m1 m2)))
            (declare (type (unsigned-byte 64) m1 m2 m))
            (if (zerop m)
                (incf cur 62)
                (let ((offset (1- (integer-length (logand m (- m))))))
                  (return-from find-3run-simd (min (+ cur offset) (+ pos 127)))))))
    (let ((limit (min max-len (+ pos 127))))
      (loop while (< cur limit)
            do (when (and (< (+ cur 2) max-len)
                          (= (aref arr cur) (aref arr (1+ cur)))
                          (= (aref arr cur) (aref arr (+ cur 2))))
                 (return-from find-3run-simd cur))
               (incf cur))
      limit)))

(declaim (inline rle-encode))
(defun rle-encode (in out in-length)
  "AVX-512 accelerated EXR RLE encoder."
  (declare (type (simple-array (unsigned-byte 8) (*)) in out)
           (type fixnum in-length)
           (optimize (speed 3) (safety 0) (debug 0)))
  (let ((op 0) (pos 0))
    (declare (type fixnum op pos))
    (loop while (< pos in-length) do
      (let ((run-len (count-run-simd in pos in-length)))
        (declare (type fixnum run-len))
        (if (>= run-len 3)
            (progn
              (setf (aref out op) (1- run-len)
                    (aref out (1+ op)) (aref in pos))
              (incf op 2)
              (incf pos run-len))
            (let* ((lit-end (min (find-3run-simd in pos in-length) (+ pos 127)))
                   (n (- lit-end pos)))
              (declare (type fixnum lit-end n))
              (when (zerop n)
                (setf lit-end (1+ pos) n 1))
              (setf (aref out op) (the (unsigned-byte 8) (- 256 n)))
              (incf op)
              (loop for i fixnum from pos below lit-end do
                (setf (aref out op) (aref in i))
                (incf op))
              (setf pos lit-end)))))
    op))

(defun compress-scanline-rle (y array w pixel-type lut-b lut-g lut-r lut-b-len lut-g-len lut-r-len
                              i-buf d-buf out-buf)
  "Compress one scanline for OpenEXR RLE (Type 1)."
  (declare (type fixnum y w lut-b-len lut-g-len lut-r-len)
           (type image-buffer array)
           (type (simple-array (unsigned-byte 8) (*)) i-buf d-buf out-buf)
           (type (or null (simple-array (unsigned-byte 16) (*))) lut-b lut-g lut-r)
           (type keyword pixel-type)
           (optimize (speed 3) (safety 0) (debug 0)))
  (let* ((bytes-per-pixel (if (eq pixel-type :half) 2 4))
         (m-total (* w 3))
         (total-bytes (* m-total bytes-per-pixel))
         (half-bytes (ash total-bytes -1))
         (row-word-start (* y m-total)))
    (declare (type fixnum bytes-per-pixel m-total total-bytes half-bytes row-word-start))

    (if (eq pixel-type :half)
        (let ((lut-b-arr (the (simple-array (unsigned-byte 16) (*)) lut-b))
              (lut-g-arr (the (simple-array (unsigned-byte 16) (*)) lut-g))
              (lut-r-arr (the (simple-array (unsigned-byte 16) (*)) lut-r)))
          ;; Channel 0: B
          (let ((w-idx (+ row-word-start 2))
                (m 0)
                (m-hi m-total))
            (declare (type fixnum w-idx m m-hi))
            (dotimes (x w)
              (let* ((hits (aref array w-idx))
                     (half (if (< hits lut-b-len) (aref lut-b-arr hits) (aref lut-b-arr (1- lut-b-len)))))
                (setf (aref i-buf m) (ldb (byte 8 0) half)
                      (aref i-buf m-hi) (ldb (byte 8 8) half))
                (incf w-idx 3)
                (incf m)
                (incf m-hi))))
          ;; Channel 1: G
          (let ((w-idx (+ row-word-start 1))
                (m w)
                (m-hi (+ m-total w)))
            (declare (type fixnum w-idx m m-hi))
            (dotimes (x w)
              (let* ((hits (aref array w-idx))
                     (half (if (< hits lut-g-len) (aref lut-g-arr hits) (aref lut-g-arr (1- lut-g-len)))))
                (setf (aref i-buf m) (ldb (byte 8 0) half)
                      (aref i-buf m-hi) (ldb (byte 8 8) half))
                (incf w-idx 3)
                (incf m)
                (incf m-hi))))
          ;; Channel 2: R
          (let ((w-idx row-word-start)
                (m (* 2 w))
                (m-hi (+ m-total (* 2 w))))
            (declare (type fixnum w-idx m m-hi))
            (dotimes (x w)
              (let* ((hits (aref array w-idx))
                     (half (if (< hits lut-r-len) (aref lut-r-arr hits) (aref lut-r-arr (1- lut-r-len)))))
                (setf (aref i-buf m) (ldb (byte 8 0) half)
                      (aref i-buf m-hi) (ldb (byte 8 8) half))
                (incf w-idx 3)
                (incf m)
                (incf m-hi)))))
        ;; 32-bit FLOAT interleave
        (let ((w-idx (+ row-word-start 2))
              (m 0))
          (declare (type fixnum w-idx m))
          (dotimes (x w)
            (let* ((hits (aref array w-idx))
                   (bits (sb-kernel:single-float-bits (float hits 1.0f0)))
                   (idx (* 2 m)))
              (setf (aref i-buf idx) (ldb (byte 8 0) bits)
                    (aref i-buf (1+ idx)) (ldb (byte 8 16) bits)
                    (aref i-buf (+ half-bytes idx)) (ldb (byte 8 8) bits)
                    (aref i-buf (+ half-bytes (1+ idx))) (ldb (byte 8 24) bits)))
            (incf w-idx 3)
            (incf m))
          (let ((w-idx (+ row-word-start 1))
                (m w))
            (declare (type fixnum w-idx m))
            (dotimes (x w)
              (let* ((hits (aref array w-idx))
                     (bits (sb-kernel:single-float-bits (float hits 1.0f0)))
                     (idx (* 2 m)))
                (setf (aref i-buf idx) (ldb (byte 8 0) bits)
                      (aref i-buf (1+ idx)) (ldb (byte 8 16) bits)
                      (aref i-buf (+ half-bytes idx)) (ldb (byte 8 8) bits)
                      (aref i-buf (+ half-bytes (1+ idx))) (ldb (byte 8 24) bits)))
              (incf w-idx 3)
              (incf m)))
          (let ((w-idx row-word-start)
                (m (* 2 w)))
            (declare (type fixnum w-idx m))
            (dotimes (x w)
              (let* ((hits (aref array w-idx))
                     (bits (sb-kernel:single-float-bits (float hits 1.0f0)))
                     (idx (* 2 m)))
                (setf (aref i-buf idx) (ldb (byte 8 0) bits)
                      (aref i-buf (1+ idx)) (ldb (byte 8 16) bits)
                      (aref i-buf (+ half-bytes idx)) (ldb (byte 8 8) bits)
                      (aref i-buf (+ half-bytes (1+ idx))) (ldb (byte 8 24) bits)))
              (incf w-idx 3)
              (incf m)))))

    ;; Predictor pass
    (setf (aref d-buf 0) (aref i-buf 0))
    (loop for idx fixnum from 1 below total-bytes do
      (setf (aref d-buf idx)
            (the (unsigned-byte 8)
                 (logand (+ (- (aref i-buf idx) (aref i-buf (1- idx))) 128) #xff))))

    ;; RLE pass
    (let ((out-len (rle-encode d-buf out-buf total-bytes)))
      (let ((res (make-array out-len :element-type '(unsigned-byte 8))))
        (replace res out-buf :end2 out-len)
        res))))

(defun compress-scanline-raw (y array w pixel-type lut-b lut-g lut-r lut-b-len lut-g-len lut-r-len i-buf)
  "Uncompressed raw scanline payload (EXR Type 0: NO_COMPRESSION)."
  (declare (type fixnum y w lut-b-len lut-g-len lut-r-len)
           (type image-buffer array)
           (type (simple-array (unsigned-byte 8) (*)) i-buf)
           (type (or null (simple-array (unsigned-byte 16) (*))) lut-b lut-g lut-r)
           (type keyword pixel-type)
           (optimize (speed 3) (safety 0) (debug 0)))
  (let* ((bytes-per-pixel (if (eq pixel-type :half) 2 4))
         (m-total (* w 3))
         (total-bytes (* m-total bytes-per-pixel))
         (half-bytes (ash total-bytes -1))
         (row-word-start (* y m-total)))
    (declare (type fixnum bytes-per-pixel m-total total-bytes half-bytes row-word-start))

    (if (eq pixel-type :half)
        (let ((lut-b-arr (the (simple-array (unsigned-byte 16) (*)) lut-b))
              (lut-g-arr (the (simple-array (unsigned-byte 16) (*)) lut-g))
              (lut-r-arr (the (simple-array (unsigned-byte 16) (*)) lut-r)))
          ;; Channel 0: B
          (let ((w-idx (+ row-word-start 2))
                (m 0)
                (m-hi m-total))
            (declare (type fixnum w-idx m m-hi))
            (dotimes (x w)
              (let* ((hits (aref array w-idx))
                     (half (if (< hits lut-b-len) (aref lut-b-arr hits) (aref lut-b-arr (1- lut-b-len)))))
                (setf (aref i-buf m) (ldb (byte 8 0) half)
                      (aref i-buf m-hi) (ldb (byte 8 8) half))
                (incf w-idx 3)
                (incf m)
                (incf m-hi))))
          ;; Channel 1: G
          (let ((w-idx (+ row-word-start 1))
                (m w)
                (m-hi (+ m-total w)))
            (declare (type fixnum w-idx m m-hi))
            (dotimes (x w)
              (let* ((hits (aref array w-idx))
                     (half (if (< hits lut-g-len) (aref lut-g-arr hits) (aref lut-g-arr (1- lut-g-len)))))
                (setf (aref i-buf m) (ldb (byte 8 0) half)
                      (aref i-buf m-hi) (ldb (byte 8 8) half))
                (incf w-idx 3)
                (incf m)
                (incf m-hi))))
          ;; Channel 2: R
          (let ((w-idx row-word-start)
                (m (* 2 w))
                (m-hi (+ m-total (* 2 w))))
            (declare (type fixnum w-idx m m-hi))
            (dotimes (x w)
              (let* ((hits (aref array w-idx))
                     (half (if (< hits lut-r-len) (aref lut-r-arr hits) (aref lut-r-arr (1- lut-r-len)))))
                (setf (aref i-buf m) (ldb (byte 8 0) half)
                      (aref i-buf m-hi) (ldb (byte 8 8) half))
                (incf w-idx 3)
                (incf m)
                (incf m-hi)))))
        ;; 32-bit FLOAT interleave
        (let ((w-idx (+ row-word-start 2))
              (m 0))
          (declare (type fixnum w-idx m))
          (dotimes (x w)
            (let* ((hits (aref array w-idx))
                   (bits (sb-kernel:single-float-bits (float hits 1.0f0)))
                   (idx (* 2 m)))
              (setf (aref i-buf idx) (ldb (byte 8 0) bits)
                    (aref i-buf (1+ idx)) (ldb (byte 8 16) bits)
                    (aref i-buf (+ half-bytes idx)) (ldb (byte 8 8) bits)
                    (aref i-buf (+ half-bytes (1+ idx))) (ldb (byte 8 24) bits)))
            (incf w-idx 3)
            (incf m))
          (let ((w-idx (+ row-word-start 1))
                (m w))
            (declare (type fixnum w-idx m))
            (dotimes (x w)
              (let* ((hits (aref array w-idx))
                     (bits (sb-kernel:single-float-bits (float hits 1.0f0)))
                     (idx (* 2 m)))
                (setf (aref i-buf idx) (ldb (byte 8 0) bits)
                      (aref i-buf (1+ idx)) (ldb (byte 8 16) bits)
                      (aref i-buf (+ half-bytes idx)) (ldb (byte 8 8) bits)
                      (aref i-buf (+ half-bytes (1+ idx))) (ldb (byte 8 24) bits)))
              (incf w-idx 3)
              (incf m)))
          (let ((w-idx row-word-start)
                (m (* 2 w)))
            (declare (type fixnum w-idx m))
            (dotimes (x w)
              (let* ((hits (aref array w-idx))
                     (bits (sb-kernel:single-float-bits (float hits 1.0f0)))
                     (idx (* 2 m)))
                (setf (aref i-buf idx) (ldb (byte 8 0) bits)
                      (aref i-buf (1+ idx)) (ldb (byte 8 16) bits)
                      (aref i-buf (+ half-bytes idx)) (ldb (byte 8 8) bits)
                      (aref i-buf (+ half-bytes (1+ idx))) (ldb (byte 8 24) bits)))
              (incf w-idx 3)
              (incf m)))))

    ;; Raw uncompressed payload (no predictor)
    (let ((res (make-array total-bytes :element-type '(unsigned-byte 8))))
      (replace res i-buf :end2 total-bytes)
      res)))

;;; ============================================================================
;;; OpenEXR Core Encoding Helpers
;;; ============================================================================

(declaim (inline float-to-half-bits))

(defun float-to-half-bits (f)
  "Fast single-float to half-precision IEEE 754 bits converter."
  (declare (type single-float f)
           (optimize (speed 3) (safety 0) (debug 0) (space 0)))
  (let* ((bits (sb-kernel:single-float-bits f))
         (sign (ash (ldb (byte 1 31) bits) 15))
         (exponent (ldb (byte 8 23) bits))
         (mantissa (ldb (byte 23 0) bits)))
    (declare (type (unsigned-byte 32) bits)
             (type (unsigned-byte 16) sign)
             (type (unsigned-byte 8) exponent)
             (type (unsigned-byte 23) mantissa))
    (cond
      ((zerop exponent) sign)
      ((= exponent 255)
       (if (zerop mantissa)
           (logior sign #x7c00) ; Inf
           (logior sign #x7e00))) ; NaN
      (t
       (let ((new-exp (- (the fixnum exponent) 112)))
         (declare (type fixnum new-exp))
         (cond
           ((>= new-exp 31)
            (logior sign #x7c00))
           ((> new-exp 0)
            (logior sign
                    (the (unsigned-byte 16) (ash new-exp 10))
                    (the (unsigned-byte 16) (ash mantissa -13))))
           ((>= new-exp -10)
            (let* ((sig (logior mantissa #x800000))
                   (shift (- 14 new-exp))
                   (sub-m (ash sig (- shift))))
              (logior sign (the (unsigned-byte 16) sub-m))))
           (t sign)))))))

(defun make-exr-header (width height dw-xmin dw-ymin dw-xmax dw-ymax has-alpha pixel-type compression-type)
  "Construct spec-compliant OpenEXR header attributes for RGB image."
  (let ((out (make-array 1024 :element-type '(unsigned-byte 8) :adjustable t :fill-pointer 0)))
    (labels ((write-bytes (vector)
               (loop for byte across vector do (vector-push-extend byte out)))
             (write-null-str (str)
               (loop for char across str do (vector-push-extend (char-code char) out))
               (vector-push-extend 0 out))
             (write-u32 (val)
               (vector-push-extend (ldb (byte 8 0) val) out)
               (vector-push-extend (ldb (byte 8 8) val) out)
               (vector-push-extend (ldb (byte 8 16) val) out)
               (vector-push-extend (ldb (byte 8 24) val) out))
             (write-u8 (val)
               (vector-push-extend val out))
             (write-attr (name type val-bytes)
               (write-null-str name)
               (write-null-str type)
               (write-u32 (length val-bytes))
               (write-bytes val-bytes)))
      ;; Magic number & version
      (write-bytes #(118 47 49 1))
      (write-bytes #(2 0 0 0))
      ;; Channels (B, G, R)
      (let ((chlist (make-array 64 :element-type '(unsigned-byte 8) :adjustable t :fill-pointer 0)))
        (labels ((add-ch (ch-name)
                   (loop for char across ch-name do (vector-push-extend (char-code char) chlist))
                   (vector-push-extend 0 chlist)
                   (let ((pt (if (eq pixel-type :half) 1 2)))
                     (vector-push-extend (ldb (byte 8 0) pt) chlist)
                     (vector-push-extend (ldb (byte 8 8) pt) chlist)
                     (vector-push-extend (ldb (byte 8 16) pt) chlist)
                     (vector-push-extend (ldb (byte 8 24) pt) chlist))
                   (vector-push-extend 0 chlist)
                   (vector-push-extend 0 chlist)
                   (vector-push-extend 0 chlist)
                   (vector-push-extend 0 chlist)
                   (dotimes (i 4) (vector-push-extend (if (= i 0) 1 0) chlist))
                   (dotimes (i 4) (vector-push-extend (if (= i 0) 1 0) chlist))))
          (when has-alpha (add-ch "A"))
          (add-ch "B")
          (add-ch "G")
          (add-ch "R")
          (vector-push-extend 0 chlist))
        (write-attr "channels" "chlist" chlist))
      ;; Compression: 0 = None, 1 = RLE
      (write-attr "compression" "compression"
                  (vector (if (member compression-type '(:none :no-compression)) 0 1)))
      ;; DataWindow
      (let ((win (make-array 16 :element-type '(unsigned-byte 8))))
        (labels ((pack-win (idx val)
                   (setf (aref win (* idx 4)) (ldb (byte 8 0) val)
                         (aref win (+ (* idx 4) 1)) (ldb (byte 8 8) val)
                         (aref win (+ (* idx 4) 2)) (ldb (byte 8 16) val)
                         (aref win (+ (* idx 4) 3)) (ldb (byte 8 24) val))))
          (pack-win 0 dw-xmin)
          (pack-win 1 dw-ymin)
          (pack-win 2 dw-xmax)
          (pack-win 3 dw-ymax))
        (write-attr "dataWindow" "box2i" win))
      ;; DisplayWindow
      (let ((win (make-array 16 :element-type '(unsigned-byte 8))))
        (labels ((pack-win (idx val)
                   (setf (aref win (* idx 4)) (ldb (byte 8 0) val)
                         (aref win (+ (* idx 4) 1)) (ldb (byte 8 8) val)
                         (aref win (+ (* idx 4) 2)) (ldb (byte 8 16) val)
                         (aref win (+ (* idx 4) 3)) (ldb (byte 8 24) val))))
          (pack-win 0 0)
          (pack-win 1 0)
          (pack-win 2 (1- width))
          (pack-win 3 (1- height)))
        (write-attr "displayWindow" "box2i" win))
      ;; LineOrder
      (write-attr "lineOrder" "lineOrder" #(0))
      ;; PixelAspectRatio
      (let ((bytes (make-array 4 :element-type '(unsigned-byte 8)))
            (bits (sb-kernel:single-float-bits 1.0f0)))
        (setf (aref bytes 0) (ldb (byte 8 0) bits)
              (aref bytes 1) (ldb (byte 8 8) bits)
              (aref bytes 2) (ldb (byte 8 16) bits)
              (aref bytes 3) (ldb (byte 8 24) bits))
        (write-attr "pixelAspectRatio" "float" bytes))
      ;; ScreenWindowCenter
      (write-attr "screenWindowCenter" "v2f" #(0 0 0 0 0 0 0 0))
      ;; ScreenWindowWidth
      (let ((bytes (make-array 4 :element-type '(unsigned-byte 8)))
            (bits (sb-kernel:single-float-bits 1.0f0)))
        (setf (aref bytes 0) (ldb (byte 8 0) bits)
              (aref bytes 1) (ldb (byte 8 8) bits)
              (aref bytes 2) (ldb (byte 8 16) bits)
              (aref bytes 3) (ldb (byte 8 24) bits))
        (write-attr "screenWindowWidth" "float" bytes))
      ;; Header Terminator
      (write-u8 0)
      out)))


;;; ============================================================================
;;; Intel Open Image Denoise (OIDN) Delegation (see denoise.lisp)
;;; ============================================================================

(defun denoise-buffer (array width height &key (scale 0.001f0) (threads 16) (hdr t))
  "Filter hit-count ARRAY using Intel Open Image Denoise (OIDN) in its own dedicated
worker thread on default GPU (CUDA). Loads denoise.lisp on demand if not present."
  (unless (find-package :oidn)
    (let ((denoise-path (merge-pathnames "denoise.lisp"
                                         (or *load-truename* *compile-file-truename* *default-pathname-defaults*))))
      (if (probe-file denoise-path)
          (load denoise-path)
          (error "denoise.lisp not found at ~A" denoise-path))))
  (funcall (find-symbol "DENOISE-BUFFER-THREAD" :oidn)
           array width height :scale scale :threads threads :hdr hdr))

(defun export-exr (filename array w h
                   &key (threads 16)
                        (pixel-type :half)
                        (tone-mapping :logarithmic)
                        (compression :rle)
                        (denoise nil))
  "Export the accumulated image buffer to an OpenEXR file.
When :denoise is true, filters the beauty buffer using Intel Open Image Denoise (OIDN)
before tone mapping and scanline compression.
Supports :rle (default, EXR Type 1) and :none / :no-compression (Type 0)."
  (assert (member pixel-type '(:half :float)) (pixel-type)
          "pixel-type must be :half or :float")
  (assert (member compression '(:rle :none :no-compression)) (compression)
          "compression must be :rle or :none")
  (when (eq pixel-type :float)
    (setf tone-mapping nil))
  (when (eq pixel-type :half)
    (assert (member tone-mapping '(:logarithmic :linear)) (tone-mapping)
            "tone-mapping must be :logarithmic or :linear for :half"))
  (format t "Exporting EXR (~A, ~A, Denoise: ~A): ~A (~D x ~D)...~%"
          pixel-type compression (if denoise "ENABLED (GPU OIDN)" "DISABLED (Bypassed)") filename w h)
  (finish-output)
  (let* ((active-array (if denoise
                           (denoise-buffer array w h :threads (min threads 16))
                           array))
         (num-blocks h)
         (header (make-exr-header w h 0 0 (1- w) (1- h) nil pixel-type compression))
         (compressed-payloads (make-array num-blocks))
         (lut-b nil)
         (lut-g nil)
         (lut-r nil)
         (max-b 1)
         (max-g 1)
         (max-r 1))
    (declare (type fixnum num-blocks w h)
             (type image-buffer active-array))

    ;; Precompute LUTs for :half mode
    (when (eq pixel-type :half)
      (multiple-value-setq (max-r max-g max-b)
        (find-channel-maximums active-array (* w h 3) threads))
      (format t "Channel maximum hits — Red: ~D, Green: ~D, Blue: ~D~%" max-r max-g max-b)
      (finish-output)
      (flet ((build-lut (max-val)
               (let* ((lut-len (min (1+ max-val) 2000000))
                      (lut (make-array lut-len :element-type '(unsigned-byte 16))))
                 (cond
                   ((eq tone-mapping :logarithmic)
                    (let ((scale (/ 1.0f0 (log (+ 1.0f0 (float max-val 1.0f0))))))
                      (dotimes (i lut-len)
                        (let* ((norm (* (log (+ 1.0f0 (float i 1.0f0))) scale))
                               (f-val (float norm 1.0f0))
                               (h-val (float-to-half-bits f-val)))
                          (setf (aref lut i) h-val)))))
                   ((eq tone-mapping :linear)
                    (let ((scale (/ 1.0f0 (max 1.0f0 (float max-val 1.0f0)))))
                      (dotimes (i lut-len)
                        (let* ((norm (* (float i 1.0f0) scale))
                               (f-val (float norm 1.0f0))
                               (h-val (float-to-half-bits f-val)))
                          (setf (aref lut i) h-val))))))
                 lut)))
        (setf lut-b (build-lut max-b)
              lut-g (build-lut max-g)
              lut-r (build-lut max-r))))

    ;; Parallel Scanline Block Compression
    (let* ((bytes-per-pixel (if (eq pixel-type :half) 2 4))
           (max-block-bytes (* w 3 bytes-per-pixel))
           (block-counter (make-array 1 :element-type 'sb-ext:word :initial-element 0))
           (num-workers (max 1 (min threads num-blocks)))
           (lut-b-len (length lut-b))
           (lut-g-len (length lut-g))
           (lut-r-len (length lut-r)))
      (let ((worker-list
              (loop for worker-id below num-workers
                    collect
                    (sb-thread:make-thread
                     (lambda ()
                       (let ((i-buf (make-array max-block-bytes :element-type '(unsigned-byte 8)))
                             (d-buf (make-array max-block-bytes :element-type '(unsigned-byte 8)))
                             (out-buf (make-array (+ max-block-bytes 4096) :element-type '(unsigned-byte 8))))
                         (loop
                           (let ((b (sb-ext:atomic-incf (aref block-counter 0))))
                             (when (>= b num-blocks) (return))
                             (setf (aref compressed-payloads b)
                                   (if (eq compression :rle)
                                       (compress-scanline-rle b active-array w pixel-type
                                                              lut-b lut-g lut-r lut-b-len lut-g-len lut-r-len
                                                              i-buf d-buf out-buf)
                                       (compress-scanline-raw b active-array w pixel-type
                                                              lut-b lut-g lut-r lut-b-len lut-g-len lut-r-len
                                                              i-buf)))))))
                     :name (format nil "exr-worker-~D" worker-id)))))
        (mapc #'sb-thread:join-thread worker-list)))

    ;; Write file
    (with-open-file (stream filename
                            :direction :output
                            :if-exists :supersede
                            :if-does-not-exist :create
                            :element-type '(unsigned-byte 8))
      (write-sequence header stream)
      ;; Bulk write offset table (64-bit LE integers)
      (let ((table (make-array (* num-blocks 8) :element-type '(unsigned-byte 8)))
            (current-offset (+ (length header) (* num-blocks 8))))
        (sb-sys:with-pinned-objects (table)
          (let ((sap (sb-sys:vector-sap table)))
            (dotimes (b num-blocks)
              (setf (sb-sys:sap-ref-64 sap (* b 8)) current-offset)
              (incf current-offset (+ 8 (length (aref compressed-payloads b)))))))
        (write-sequence table stream))
      ;; Bulk write scanline blocks (4-byte y + 4-byte len + payload)
      (let ((hdr (make-array 8 :element-type '(unsigned-byte 8))))
        (sb-sys:with-pinned-objects (hdr)
          (let ((sap (sb-sys:vector-sap hdr)))
            (dotimes (b num-blocks)
              (let ((payload (aref compressed-payloads b)))
                (setf (sb-sys:sap-ref-32 sap 0) b
                      (sb-sys:sap-ref-32 sap 4) (length payload))
                (write-sequence hdr stream)
                (write-sequence payload stream)))))))
    (format t "Export complete: ~A (~A, ~A) written successfully.~%" filename pixel-type compression)
    (finish-output)))

(defun render (&key (threads 16)
                    (samples-per-thread 5000)
                    (output *default-output-filename*)
                    (width +default-width+)
                    (height +default-height+)
                    (pixel-type :half)
                    (tone-mapping :logarithmic)
                    (precision :single)
                    (compression :rle)
                    (cycle-detection t)
                    (denoise nil))
  "Main entry point: Parallel AVX-512 Buddhabrot renderer with OpenEXR export and cycle detection."
  (declare (type fixnum threads samples-per-thread width height)
           (type keyword pixel-type precision compression)
           (type boolean cycle-detection))
  (let* ((img-width width)
         (img-height height)
         (total-pixels (the fixnum (* (the fixnum (* img-width img-height)) 3)))
         (array (if (and (= img-width +default-width+) (= img-height +default-height+))
                    *color-array*
                    (make-array total-pixels :element-type '(unsigned-byte 32) :initial-element 0)))
         (total-target (the fixnum (* threads samples-per-thread))))
    (declare (type fixnum img-width img-height total-pixels total-target)
             (type image-buffer array))
    ;; Parallel-zero buffer before starting the compute phase clock
    (parallel-fill array total-pixels threads)
    (format t "========================================================================~%")
    (format t "Buddhabrot AVX-512: ~D threads, ~D orbits/thread (~D total target orbits)~%"
            threads samples-per-thread total-target)
    (format t "Resolution: ~D x ~D (~,2F Megapixels)~%"
            img-width img-height (/ (float (* img-width img-height) 1d0) 1000000.0d0))
    (format t "Engine: AVX-512 ~A (~A lanes), Cycle Detection: ~A~%"
            (if (eq precision :double) "f64.8 (Double-Precision)" "f32.16 (Single-Precision)")
            (if (eq precision :double) 8 16)
            (if cycle-detection "ENABLED (K=2520)" "DISABLED"))
    (format t "Output: ~A, Pixel Type: ~A, Compression: ~A, Tone Mapping: ~A, Denoise: ~A~%"
            output pixel-type compression (if (eq pixel-type :half) tone-mapping "NONE (scene-referred)")
            (if denoise "ENABLED (GPU OIDN)" "DISABLED (Bypassed)"))
    (format t "Symmetry optimization: ENABLED (2x throughput)~%")
    (format t "========================================================================~%")
    (finish-output)
    (let ((start-time (get-internal-real-time)))
      (declare (type (unsigned-byte 64) start-time))
      (let ((thread-list
              (loop for i of-type fixnum below threads
                    collect
                    (let* ((thread-id i)
                           (seed (logand #xFFFFFFFFFFFFFFFF
                                         (+ (* (get-internal-real-time) 1000003)
                                            (* thread-id 982451653)
                                            #x123456789ABCDEF)))
                            (rng (make-fast-rng seed)))
                      (sb-thread:make-thread
                       (lambda ()
                         (if (eq precision :double)
                             (simd-worker-f64 samples-per-thread rng array img-width img-height cycle-detection)
                             (simd-worker-f32 samples-per-thread rng array img-width img-height cycle-detection)))
                       :name (format nil "buddha-worker-~D" thread-id))))))
        (mapc #'sb-thread:join-thread thread-list))
      (let* ((now (get-internal-real-time))
             (elapsed-sec (/ (float (- now start-time) 1d0)
                             (float internal-time-units-per-second 1d0)))
             (effective-orbits (float (* total-target 2) 1d0)))
        (format t "Computation completed in ~,3F seconds (~,1F effective orbits/sec).~%"
                elapsed-sec (/ effective-orbits (max 0.001d0 elapsed-sec)))))
    ;; Save to global for draw backward-compatibility
    (setf *color-array* array)
    (export-exr output array img-width img-height
                :threads threads
                :pixel-type pixel-type
                :compression compression
                :tone-mapping (if (eq pixel-type :half) tone-mapping nil)
                :denoise denoise)))

(defun main (&optional (arg1 16) (arg2 5000) (arg3 nil) (output *default-output-filename*))
  "Flexible entry point for REPL and Makefile:
   - (main) -> 16 threads, 5,000 samples/thread
   - (main 10000) -> detects 10,000 > 256, runs 10,000 samples/thread on 16 threads
   - (main 16 10000) -> 16 threads, 10,000 samples/thread
   - (main 16 10000 t) -> 16 threads, 10,000 samples/thread, denoised"
  (let ((threads (if (and (numberp arg1) (<= arg1 256)) arg1 16))
        (samples (cond
                   ((and (numberp arg1) (> arg1 256)) arg1)
                   ((numberp arg2) arg2)
                   (t 5000)))
        (denoise (cond
                   ((typep arg2 'boolean) arg2)
                   ((typep arg3 'boolean) arg3)
                   (t nil)))
        (out (cond
               ((stringp arg2) arg2)
               ((stringp arg3) arg3)
               (t output))))
    (render :threads threads :samples-per-thread samples :output out :denoise denoise)))

(defun main-denoise (&optional (arg1 16) (arg2 5000) (output "buddhabrot-denoised.exr"))
  "Entry point for Makefile targets with Intel Open Image Denoise (OIDN) enabled on GPU."
  (let ((threads (if (and (numberp arg1) (<= arg1 256)) arg1 16))
        (samples (cond
                   ((and (numberp arg1) (> arg1 256)) arg1)
                   ((numberp arg2) arg2)
                   (t 5000)))
        (out (cond
               ((stringp arg2) arg2)
               (t output))))
    (render :threads threads :samples-per-thread samples :output out :denoise t)))

(defun run-thread (&optional (num-threads 16))
  "Backward-compatible alias for RUN-THREAD."
  (main num-threads))

(defun draw (&key (output *default-output-filename*) (pixel-type :half) (tone-mapping :logarithmic) (denoise nil))
  "Backward-compatible alias for exporting the current *color-array* to OpenEXR, optionally with OIDN denoising."
  (export-exr output *color-array* +default-width+ +default-height+
              :pixel-type pixel-type
              :tone-mapping (if (eq pixel-type :half) tone-mapping nil)
              :denoise denoise))
