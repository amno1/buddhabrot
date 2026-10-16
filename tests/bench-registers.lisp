;;;; bench-registers.lisp — Register Saturation & ILP Benchmark for AVX-512
;;;; Compares 1x vector (16 lanes, ~10 ZMM regs)
;;;;       vs 2x vectors (32 lanes, ~19 ZMM regs)
;;;;       vs 3x vectors (48 lanes, ~26 ZMM regs)

(in-package :cl-user)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (require :sb-simd))

;; Locate and load buddha-512 (prefer FASL, fallback to .lisp)
(let* ((script-path (or *load-pathname* *compile-file-pathname* (truename ".")))
       (tests-dir (make-pathname :directory (pathname-directory (truename script-path))))
       ;; If script is inside tests/, root-dir is parent; otherwise current directory
       (root-dir (if (equal (car (last (pathname-directory tests-dir))) "tests")
                     (make-pathname :directory (butlast (pathname-directory tests-dir)))
                     tests-dir))
       (fasl-path (merge-pathnames "buddha-512.fasl" root-dir))
       (lisp-path (merge-pathnames "buddha-512.lisp" root-dir)))
  (cond
    ((probe-file fasl-path) (load fasl-path))
    ((probe-file lisp-path) (load lisp-path))
    (t (error "Could not locate buddha-512.fasl or buddha-512.lisp in ~A" root-dir))))

(defpackage #:bench-regs
  (:use #:cl #:sb-simd-avx512bw))

(in-package :bench-regs)

(setf *read-default-float-format* 'double-float)
(declaim (optimize (speed 3) (safety 0) (debug 0)))
(sb-int:set-floating-point-modes :traps nil)

;;; ============================================================================
;;; Part 1: Raw Mandelbrot Step Iteration (Pure Math in Registers, No Memory/Branch)
;;; ============================================================================

(defun bench-raw-1x (iterations)
  (declare (type fixnum iterations) (optimize (speed 3) (safety 0)))
  (let ((zr (sb-simd-avx512f:f32.16-broadcast 0.0f0))
        (zi (sb-simd-avx512f:f32.16-broadcast 0.0f0))
        (cr (sb-simd-avx512f:f32.16-broadcast -0.5f0))
        (ci (sb-simd-avx512f:f32.16-broadcast 0.1f0)))
    (declare (type sb-simd-avx512f:f32.16 zr zi cr ci))
    (dotimes (i iterations)
      (let* ((zr2 (sb-simd-avx512f:f32.16* zr zr))
             (zi2 (sb-simd-avx512f:f32.16* zi zi))
             (zri (sb-simd-avx512f:f32.16* zr zi)))
        (setf zr (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16- zr2 zi2) cr)
              zi (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16+ zri zri) ci))))
    zr))

(defun bench-raw-2x (iterations)
  (declare (type fixnum iterations) (optimize (speed 3) (safety 0)))
  (let ((zr0 (sb-simd-avx512f:f32.16-broadcast 0.0f0))
        (zi0 (sb-simd-avx512f:f32.16-broadcast 0.0f0))
        (cr0 (sb-simd-avx512f:f32.16-broadcast -0.5f0))
        (ci0 (sb-simd-avx512f:f32.16-broadcast 0.1f0))
        (zr1 (sb-simd-avx512f:f32.16-broadcast 0.0f0))
        (zi1 (sb-simd-avx512f:f32.16-broadcast 0.0f0))
        (cr1 (sb-simd-avx512f:f32.16-broadcast -0.4f0))
        (ci1 (sb-simd-avx512f:f32.16-broadcast 0.15f0)))
    (declare (type sb-simd-avx512f:f32.16 zr0 zi0 cr0 ci0 zr1 zi1 cr1 ci1))
    (dotimes (i iterations)
      (let* ((zr2-0 (sb-simd-avx512f:f32.16* zr0 zr0))
             (zi2-0 (sb-simd-avx512f:f32.16* zi0 zi0))
             (zri-0 (sb-simd-avx512f:f32.16* zr0 zi0))
             (zr2-1 (sb-simd-avx512f:f32.16* zr1 zr1))
             (zi2-1 (sb-simd-avx512f:f32.16* zi1 zi1))
             (zri-1 (sb-simd-avx512f:f32.16* zr1 zi1)))
        (setf zr0 (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16- zr2-0 zi2-0) cr0)
              zi0 (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16+ zri-0 zri-0) ci0)
              zr1 (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16- zr2-1 zi2-1) cr1)
              zi1 (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16+ zri-1 zri-1) ci1))))
    (sb-simd-avx512f:f32.16+ zr0 zr1)))

(defun bench-raw-3x (iterations)
  (declare (type fixnum iterations) (optimize (speed 3) (safety 0)))
  (let ((zr0 (sb-simd-avx512f:f32.16-broadcast 0.0f0))
        (zi0 (sb-simd-avx512f:f32.16-broadcast 0.0f0))
        (cr0 (sb-simd-avx512f:f32.16-broadcast -0.5f0))
        (ci0 (sb-simd-avx512f:f32.16-broadcast 0.1f0))
        (zr1 (sb-simd-avx512f:f32.16-broadcast 0.0f0))
        (zi1 (sb-simd-avx512f:f32.16-broadcast 0.0f0))
        (cr1 (sb-simd-avx512f:f32.16-broadcast -0.4f0))
        (ci1 (sb-simd-avx512f:f32.16-broadcast 0.15f0))
        (zr2 (sb-simd-avx512f:f32.16-broadcast 0.0f0))
        (zi2 (sb-simd-avx512f:f32.16-broadcast 0.0f0))
        (cr2 (sb-simd-avx512f:f32.16-broadcast -0.3f0))
        (ci2 (sb-simd-avx512f:f32.16-broadcast 0.2f0)))
    (declare (type sb-simd-avx512f:f32.16 zr0 zi0 cr0 ci0 zr1 zi1 cr1 ci1 zr2 zi2 cr2 ci2))
    (dotimes (i iterations)
      (let* ((zr2-0 (sb-simd-avx512f:f32.16* zr0 zr0))
             (zi2-0 (sb-simd-avx512f:f32.16* zi0 zi0))
             (zri-0 (sb-simd-avx512f:f32.16* zr0 zi0))
             (zr2-1 (sb-simd-avx512f:f32.16* zr1 zr1))
             (zi2-1 (sb-simd-avx512f:f32.16* zi1 zi1))
             (zri-1 (sb-simd-avx512f:f32.16* zr1 zi1))
             (zr2-2 (sb-simd-avx512f:f32.16* zr2 zr2))
             (zi2-2 (sb-simd-avx512f:f32.16* zi2 zi2))
             (zri-2 (sb-simd-avx512f:f32.16* zr2 zi2)))
        (setf zr0 (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16- zr2-0 zi2-0) cr0)
              zi0 (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16+ zri-0 zri-0) ci0)
              zr1 (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16- zr2-1 zi2-1) cr1)
              zi1 (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16+ zri-1 zri-1) ci1)
              zr2 (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16- zr2-2 zi2-2) cr2)
              zi2 (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16+ zri-2 zri-2) ci2))))
    (sb-simd-avx512f:f32.16+ zr0 (sb-simd-avx512f:f32.16+ zr1 zr2))))

;;; ============================================================================
;;; Part 2: Real Buddhabrot Escaped Orbit Workers (Continuous Refill + Retrace)
;;; ============================================================================

(defun simd-worker-f32-2x (target-hits rng array w h &optional atomic (cycle-detection t))
  (declare (type fixnum target-hits w h)
           (type buddha::xoshiro rng)
           (type buddha::image-buffer array)
           (type boolean atomic cycle-detection)
           (optimize (speed 3) (safety 0) (debug 0)))
  (let* ((cr-arr0 (make-array 16 :element-type 'single-float))
         (ci-arr0 (make-array 16 :element-type 'single-float))
         (zr-arr0 (make-array 16 :element-type 'single-float))
         (zi-arr0 (make-array 16 :element-type 'single-float))
         (start-steps0 (make-array 16 :element-type '(unsigned-byte 32) :initial-element 0))
         (snap-zr0 (make-array 16 :element-type 'single-float :initial-element 0.0f0))
         (snap-zi0 (make-array 16 :element-type 'single-float :initial-element 0.0f0))
         (snap-valid0 (make-array 16 :element-type 'bit :initial-element 0))

         (cr-arr1 (make-array 16 :element-type 'single-float))
         (ci-arr1 (make-array 16 :element-type 'single-float))
         (zr-arr1 (make-array 16 :element-type 'single-float))
         (zi-arr1 (make-array 16 :element-type 'single-float))
         (start-steps1 (make-array 16 :element-type '(unsigned-byte 32) :initial-element 0))
         (snap-zr1 (make-array 16 :element-type 'single-float :initial-element 0.0f0))
         (snap-zi1 (make-array 16 :element-type 'single-float :initial-element 0.0f0))
         (snap-valid1 (make-array 16 :element-type 'bit :initial-element 0))

         (four (sb-simd-avx512f:f32.16-broadcast buddha::+escape-radius-sq-f32+))
         (scale-x (float (/ (float w 1d0) buddha::+image-x+) 1.0f0))
         (scale-y (float (/ (float h 1d0) buddha::+image-y+) 1.0f0))
         (px-mul scale-x)
         (py-mul scale-y)
         (x-off (* (- buddha::+start-image-x-f32+) scale-x))
         (y-off (* (- buddha::+end-image-y-f32+) scale-y))
         (hits 0)
         (global-step 0)
         (steps-since-check 0))
    (declare (type (simple-array single-float (16))
                   cr-arr0 ci-arr0 zr-arr0 zi-arr0 snap-zr0 snap-zi0
                   cr-arr1 ci-arr1 zr-arr1 zi-arr1 snap-zr1 snap-zi1)
             (type (simple-array (unsigned-byte 32) (16)) start-steps0 start-steps1)
             (type (simple-array bit (16)) snap-valid0 snap-valid1)
             (type single-float px-mul py-mul x-off y-off)
             (type fixnum hits steps-since-check)
             (type (unsigned-byte 32) global-step)
             (type sb-simd-avx512f:f32.16 four))

    (dotimes (i 16)
      (multiple-value-bind (cr ci) (buddha::generate-candidate-f32 rng)
        (setf (aref cr-arr0 i) cr (aref ci-arr0 i) ci
              (aref zr-arr0 i) cr (aref zi-arr0 i) ci
              (aref start-steps0 i) 0 (aref snap-valid0 i) 0))
      (multiple-value-bind (cr ci) (buddha::generate-candidate-f32 rng)
        (setf (aref cr-arr1 i) cr (aref ci-arr1 i) ci
              (aref zr-arr1 i) cr (aref zi-arr1 i) ci
              (aref start-steps1 i) 0 (aref snap-valid1 i) 0)))

    (let ((cr-vec0 (sb-simd-avx512f:f32.16-aref cr-arr0 0))
          (ci-vec0 (sb-simd-avx512f:f32.16-aref ci-arr0 0))
          (zr-vec0 (sb-simd-avx512f:f32.16-aref zr-arr0 0))
          (zi-vec0 (sb-simd-avx512f:f32.16-aref zi-arr0 0))
          (cr-vec1 (sb-simd-avx512f:f32.16-aref cr-arr1 0))
          (ci-vec1 (sb-simd-avx512f:f32.16-aref ci-arr1 0))
          (zr-vec1 (sb-simd-avx512f:f32.16-aref zr-arr1 0))
          (zi-vec1 (sb-simd-avx512f:f32.16-aref zi-arr1 0)))
      (declare (type sb-simd-avx512f:f32.16 cr-vec0 ci-vec0 zr-vec0 zi-vec0
                                             cr-vec1 ci-vec1 zr-vec1 zi-vec1))
      (loop while (< hits target-hits) do
        (incf global-step)
        (let* ((zr2-0 (sb-simd-avx512f:f32.16* zr-vec0 zr-vec0))
               (zi2-0 (sb-simd-avx512f:f32.16* zi-vec0 zi-vec0))
               (zri-0 (sb-simd-avx512f:f32.16* zr-vec0 zi-vec0))
               (zr2-1 (sb-simd-avx512f:f32.16* zr-vec1 zr-vec1))
               (zi2-1 (sb-simd-avx512f:f32.16* zi-vec1 zi-vec1))
               (zri-1 (sb-simd-avx512f:f32.16* zr-vec1 zi-vec1))
               (mag2-0 (sb-simd-avx512f:f32.16+ zr2-0 zi2-0))
               (mag2-1 (sb-simd-avx512f:f32.16+ zr2-1 zi2-1))
               (esc0 (sb-simd-avx512f:f32.16-mask>= mag2-0 four))
               (esc1 (sb-simd-avx512f:f32.16-mask>= mag2-1 four))
               (m0 (sb-kernel:%simd-pack-512-mask-value esc0))
               (m1 (sb-kernel:%simd-pack-512-mask-value esc1))
               (check-bailout (>= (incf steps-since-check) buddha::+cycle-detection-interval+)))
          (declare (type (unsigned-byte 16) m0 m1))
          (if (and (zerop (logior m0 m1)) (not check-bailout))
              (setf zr-vec0 (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16- zr2-0 zi2-0) cr-vec0)
                    zi-vec0 (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16+ zri-0 zri-0) ci-vec0)
                    zr-vec1 (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16- zr2-1 zi2-1) cr-vec1)
                    zi-vec1 (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16+ zri-1 zri-1) ci-vec1))
              (progn
                (when check-bailout (setf steps-since-check 0))
                (setf (sb-simd-avx512f:f32.16-aref zr-arr0 0) (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16- zr2-0 zi2-0) cr-vec0)
                      (sb-simd-avx512f:f32.16-aref zi-arr0 0) (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16+ zri-0 zri-0) ci-vec0)
                      (sb-simd-avx512f:f32.16-aref zr-arr1 0) (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16- zr2-1 zi2-1) cr-vec1)
                      (sb-simd-avx512f:f32.16-aref zi-arr1 0) (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16+ zri-1 zri-1) ci-vec1))

                ;; Vector 0
                (dotimes (i 16)
                  (let ((steps (- global-step (aref start-steps0 i))))
                    (declare (type (unsigned-byte 32) steps))
                    (cond
                      ((logbitp i m0)
                       (when (<= buddha::+min-bailout+ steps buddha::+max-bailout+)
                         (if atomic
                             (buddha::trace-symmetric-orbit-f32-atomic (aref cr-arr0 i) (aref ci-arr0 i) steps
                                                                       array w h px-mul x-off py-mul y-off)
                             (buddha::trace-symmetric-orbit-f32 (aref cr-arr0 i) (aref ci-arr0 i) steps
                                                                array w h px-mul x-off py-mul y-off))
                         (incf hits))
                       (multiple-value-bind (cr ci) (buddha::generate-candidate-f32 rng)
                         (setf (aref cr-arr0 i) cr (aref ci-arr0 i) ci
                               (aref zr-arr0 i) cr (aref zi-arr0 i) ci
                               (aref start-steps0 i) global-step (aref snap-valid0 i) 0)))
                      ((and cycle-detection check-bailout (= (aref snap-valid0 i) 1)
                            (let ((dzr (- (aref zr-arr0 i) (aref snap-zr0 i)))
                                  (dzi (- (aref zi-arr0 i) (aref snap-zi0 i))))
                              (< (+ (* dzr dzr) (* dzi dzi)) buddha::+cycle-eps-sq-f32+)))
                       (multiple-value-bind (cr ci) (buddha::generate-candidate-f32 rng)
                         (setf (aref cr-arr0 i) cr (aref ci-arr0 i) ci
                               (aref zr-arr0 i) cr (aref zi-arr0 i) ci
                               (aref start-steps0 i) global-step (aref snap-valid0 i) 0)))
                      ((>= steps buddha::+max-bailout+)
                       (multiple-value-bind (cr ci) (buddha::generate-candidate-f32 rng)
                         (setf (aref cr-arr0 i) cr (aref ci-arr0 i) ci
                               (aref zr-arr0 i) cr (aref zi-arr0 i) ci
                               (aref start-steps0 i) global-step (aref snap-valid0 i) 0))))))

                ;; Vector 1
                (dotimes (i 16)
                  (let ((steps (- global-step (aref start-steps1 i))))
                    (declare (type (unsigned-byte 32) steps))
                    (cond
                      ((logbitp i m1)
                       (when (<= buddha::+min-bailout+ steps buddha::+max-bailout+)
                         (if atomic
                             (buddha::trace-symmetric-orbit-f32-atomic (aref cr-arr1 i) (aref ci-arr1 i) steps
                                                                       array w h px-mul x-off py-mul y-off)
                             (buddha::trace-symmetric-orbit-f32 (aref cr-arr1 i) (aref ci-arr1 i) steps
                                                                array w h px-mul x-off py-mul y-off))
                         (incf hits))
                       (multiple-value-bind (cr ci) (buddha::generate-candidate-f32 rng)
                         (setf (aref cr-arr1 i) cr (aref ci-arr1 i) ci
                               (aref zr-arr1 i) cr (aref zi-arr1 i) ci
                               (aref start-steps1 i) global-step (aref snap-valid1 i) 0)))
                      ((and cycle-detection check-bailout (= (aref snap-valid1 i) 1)
                            (let ((dzr (- (aref zr-arr1 i) (aref snap-zr1 i)))
                                  (dzi (- (aref zi-arr1 i) (aref snap-zi1 i))))
                              (< (+ (* dzr dzr) (* dzi dzi)) buddha::+cycle-eps-sq-f32+)))
                       (multiple-value-bind (cr ci) (buddha::generate-candidate-f32 rng)
                         (setf (aref cr-arr1 i) cr (aref ci-arr1 i) ci
                               (aref zr-arr1 i) cr (aref zi-arr1 i) ci
                               (aref start-steps1 i) global-step (aref snap-valid1 i) 0)))
                      ((>= steps buddha::+max-bailout+)
                       (multiple-value-bind (cr ci) (buddha::generate-candidate-f32 rng)
                         (setf (aref cr-arr1 i) cr (aref ci-arr1 i) ci
                               (aref zr-arr1 i) cr (aref zi-arr1 i) ci
                               (aref start-steps1 i) global-step (aref snap-valid1 i) 0))))))

                (when (and cycle-detection check-bailout)
                  (dotimes (i 16)
                    (setf (aref snap-zr0 i) (aref zr-arr0 i) (aref snap-zi0 i) (aref zi-arr0 i) (aref snap-valid0 i) 1
                          (aref snap-zr1 i) (aref zr-arr1 i) (aref snap-zi1 i) (aref zi-arr1 i) (aref snap-valid1 i) 1)))

                (setf cr-vec0 (sb-simd-avx512f:f32.16-aref cr-arr0 0)
                      ci-vec0 (sb-simd-avx512f:f32.16-aref ci-arr0 0)
                      zr-vec0 (sb-simd-avx512f:f32.16-aref zr-arr0 0)
                      zi-vec0 (sb-simd-avx512f:f32.16-aref zi-arr0 0)
                      cr-vec1 (sb-simd-avx512f:f32.16-aref cr-arr1 0)
                      ci-vec1 (sb-simd-avx512f:f32.16-aref ci-arr1 0)
                      zr-vec1 (sb-simd-avx512f:f32.16-aref zr-arr1 0)
                      zi-vec1 (sb-simd-avx512f:f32.16-aref zi-arr1 0)))))))))

(defun simd-worker-f32-3x (target-hits rng array w h &optional atomic (cycle-detection t))
  (declare (type fixnum target-hits w h)
           (type buddha::xoshiro rng)
           (type buddha::image-buffer array)
           (type boolean atomic cycle-detection)
           (optimize (speed 3) (safety 0) (debug 0)))
  (let* ((cr-arr0 (make-array 16 :element-type 'single-float))
         (ci-arr0 (make-array 16 :element-type 'single-float))
         (zr-arr0 (make-array 16 :element-type 'single-float))
         (zi-arr0 (make-array 16 :element-type 'single-float))
         (start-steps0 (make-array 16 :element-type '(unsigned-byte 32) :initial-element 0))
         (snap-zr0 (make-array 16 :element-type 'single-float :initial-element 0.0f0))
         (snap-zi0 (make-array 16 :element-type 'single-float :initial-element 0.0f0))
         (snap-valid0 (make-array 16 :element-type 'bit :initial-element 0))

         (cr-arr1 (make-array 16 :element-type 'single-float))
         (ci-arr1 (make-array 16 :element-type 'single-float))
         (zr-arr1 (make-array 16 :element-type 'single-float))
         (zi-arr1 (make-array 16 :element-type 'single-float))
         (start-steps1 (make-array 16 :element-type '(unsigned-byte 32) :initial-element 0))
         (snap-zr1 (make-array 16 :element-type 'single-float :initial-element 0.0f0))
         (snap-zi1 (make-array 16 :element-type 'single-float :initial-element 0.0f0))
         (snap-valid1 (make-array 16 :element-type 'bit :initial-element 0))

         (cr-arr2 (make-array 16 :element-type 'single-float))
         (ci-arr2 (make-array 16 :element-type 'single-float))
         (zr-arr2 (make-array 16 :element-type 'single-float))
         (zi-arr2 (make-array 16 :element-type 'single-float))
         (start-steps2 (make-array 16 :element-type '(unsigned-byte 32) :initial-element 0))
         (snap-zr2 (make-array 16 :element-type 'single-float :initial-element 0.0f0))
         (snap-zi2 (make-array 16 :element-type 'single-float :initial-element 0.0f0))
         (snap-valid2 (make-array 16 :element-type 'bit :initial-element 0))

         (four (sb-simd-avx512f:f32.16-broadcast buddha::+escape-radius-sq-f32+))
         (scale-x (float (/ (float w 1d0) buddha::+image-x+) 1.0f0))
         (scale-y (float (/ (float h 1d0) buddha::+image-y+) 1.0f0))
         (px-mul scale-x)
         (py-mul scale-y)
         (x-off (* (- buddha::+start-image-x-f32+) scale-x))
         (y-off (* (- buddha::+end-image-y-f32+) scale-y))
         (hits 0)
         (global-step 0)
         (steps-since-check 0))
    (declare (type (simple-array single-float (16))
                   cr-arr0 ci-arr0 zr-arr0 zi-arr0 snap-zr0 snap-zi0
                   cr-arr1 ci-arr1 zr-arr1 zi-arr1 snap-zr1 snap-zi1
                   cr-arr2 ci-arr2 zr-arr2 zi-arr2 snap-zr2 snap-zi2)
             (type (simple-array (unsigned-byte 32) (16)) start-steps0 start-steps1 start-steps2)
             (type (simple-array bit (16)) snap-valid0 snap-valid1 snap-valid2)
             (type single-float px-mul py-mul x-off y-off)
             (type fixnum hits steps-since-check)
             (type (unsigned-byte 32) global-step)
             (type sb-simd-avx512f:f32.16 four))

    (dotimes (i 16)
      (multiple-value-bind (cr ci) (buddha::generate-candidate-f32 rng)
        (setf (aref cr-arr0 i) cr (aref ci-arr0 i) ci
              (aref zr-arr0 i) cr (aref zi-arr0 i) ci
              (aref start-steps0 i) 0 (aref snap-valid0 i) 0))
      (multiple-value-bind (cr ci) (buddha::generate-candidate-f32 rng)
        (setf (aref cr-arr1 i) cr (aref ci-arr1 i) ci
              (aref zr-arr1 i) cr (aref zi-arr1 i) ci
              (aref start-steps1 i) 0 (aref snap-valid1 i) 0))
      (multiple-value-bind (cr ci) (buddha::generate-candidate-f32 rng)
        (setf (aref cr-arr2 i) cr (aref ci-arr2 i) ci
              (aref zr-arr2 i) cr (aref zi-arr2 i) ci
              (aref start-steps2 i) 0 (aref snap-valid2 i) 0)))

    (let ((cr-vec0 (sb-simd-avx512f:f32.16-aref cr-arr0 0))
          (ci-vec0 (sb-simd-avx512f:f32.16-aref ci-arr0 0))
          (zr-vec0 (sb-simd-avx512f:f32.16-aref zr-arr0 0))
          (zi-vec0 (sb-simd-avx512f:f32.16-aref zi-arr0 0))
          (cr-vec1 (sb-simd-avx512f:f32.16-aref cr-arr1 0))
          (ci-vec1 (sb-simd-avx512f:f32.16-aref ci-arr1 0))
          (zr-vec1 (sb-simd-avx512f:f32.16-aref zr-arr1 0))
          (zi-vec1 (sb-simd-avx512f:f32.16-aref zi-arr1 0))
          (cr-vec2 (sb-simd-avx512f:f32.16-aref cr-arr2 0))
          (ci-vec2 (sb-simd-avx512f:f32.16-aref ci-arr2 0))
          (zr-vec2 (sb-simd-avx512f:f32.16-aref zr-arr2 0))
          (zi-vec2 (sb-simd-avx512f:f32.16-aref zi-arr2 0)))
      (declare (type sb-simd-avx512f:f32.16 cr-vec0 ci-vec0 zr-vec0 zi-vec0
                                             cr-vec1 ci-vec1 zr-vec1 zi-vec1
                                             cr-vec2 ci-vec2 zr-vec2 zi-vec2))
      (loop while (< hits target-hits) do
        (incf global-step)
        (let* ((zr2-0 (sb-simd-avx512f:f32.16* zr-vec0 zr-vec0))
               (zi2-0 (sb-simd-avx512f:f32.16* zi-vec0 zi-vec0))
               (zri-0 (sb-simd-avx512f:f32.16* zr-vec0 zi-vec0))
               (zr2-1 (sb-simd-avx512f:f32.16* zr-vec1 zr-vec1))
               (zi2-1 (sb-simd-avx512f:f32.16* zi-vec1 zi-vec1))
               (zri-1 (sb-simd-avx512f:f32.16* zr-vec1 zi-vec1))
               (zr2-2 (sb-simd-avx512f:f32.16* zr-vec2 zr-vec2))
               (zi2-2 (sb-simd-avx512f:f32.16* zi-vec2 zi-vec2))
               (zri-2 (sb-simd-avx512f:f32.16* zr-vec2 zi-vec2))
               (mag2-0 (sb-simd-avx512f:f32.16+ zr2-0 zi2-0))
               (mag2-1 (sb-simd-avx512f:f32.16+ zr2-1 zi2-1))
               (mag2-2 (sb-simd-avx512f:f32.16+ zr2-2 zi2-2))
               (esc0 (sb-simd-avx512f:f32.16-mask>= mag2-0 four))
               (esc1 (sb-simd-avx512f:f32.16-mask>= mag2-1 four))
               (esc2 (sb-simd-avx512f:f32.16-mask>= mag2-2 four))
               (m0 (sb-kernel:%simd-pack-512-mask-value esc0))
               (m1 (sb-kernel:%simd-pack-512-mask-value esc1))
               (m2 (sb-kernel:%simd-pack-512-mask-value esc2))
               (check-bailout (>= (incf steps-since-check) buddha::+cycle-detection-interval+)))
          (declare (type (unsigned-byte 16) m0 m1 m2))
          (if (and (zerop (logior m0 m1 m2)) (not check-bailout))
              (setf zr-vec0 (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16- zr2-0 zi2-0) cr-vec0)
                    zi-vec0 (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16+ zri-0 zri-0) ci-vec0)
                    zr-vec1 (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16- zr2-1 zi2-1) cr-vec1)
                    zi-vec1 (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16+ zri-1 zri-1) ci-vec1)
                    zr-vec2 (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16- zr2-2 zi2-2) cr-vec2)
                    zi-vec2 (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16+ zri-2 zri-2) ci-vec2))
              (progn
                (when check-bailout (setf steps-since-check 0))
                (setf (sb-simd-avx512f:f32.16-aref zr-arr0 0) (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16- zr2-0 zi2-0) cr-vec0)
                      (sb-simd-avx512f:f32.16-aref zi-arr0 0) (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16+ zri-0 zri-0) ci-vec0)
                      (sb-simd-avx512f:f32.16-aref zr-arr1 0) (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16- zr2-1 zi2-1) cr-vec1)
                      (sb-simd-avx512f:f32.16-aref zi-arr1 0) (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16+ zri-1 zri-1) ci-vec1)
                      (sb-simd-avx512f:f32.16-aref zr-arr2 0) (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16- zr2-2 zi2-2) cr-vec2)
                      (sb-simd-avx512f:f32.16-aref zi-arr2 0) (sb-simd-avx512f:f32.16+ (sb-simd-avx512f:f32.16+ zri-2 zri-2) ci-vec2))

                ;; Vector 0
                (dotimes (i 16)
                  (let ((steps (- global-step (aref start-steps0 i))))
                    (declare (type (unsigned-byte 32) steps))
                    (cond
                      ((logbitp i m0)
                       (when (<= buddha::+min-bailout+ steps buddha::+max-bailout+)
                         (if atomic
                             (buddha::trace-symmetric-orbit-f32-atomic (aref cr-arr0 i) (aref ci-arr0 i) steps
                                                                       array w h px-mul x-off py-mul y-off)
                             (buddha::trace-symmetric-orbit-f32 (aref cr-arr0 i) (aref ci-arr0 i) steps
                                                                array w h px-mul x-off py-mul y-off))
                         (incf hits))
                       (multiple-value-bind (cr ci) (buddha::generate-candidate-f32 rng)
                         (setf (aref cr-arr0 i) cr (aref ci-arr0 i) ci
                               (aref zr-arr0 i) cr (aref zi-arr0 i) ci
                               (aref start-steps0 i) global-step (aref snap-valid0 i) 0)))
                      ((and cycle-detection check-bailout (= (aref snap-valid0 i) 1)
                            (let ((dzr (- (aref zr-arr0 i) (aref snap-zr0 i)))
                                  (dzi (- (aref zi-arr0 i) (aref snap-zi0 i))))
                              (< (+ (* dzr dzr) (* dzi dzi)) buddha::+cycle-eps-sq-f32+)))
                       (multiple-value-bind (cr ci) (buddha::generate-candidate-f32 rng)
                         (setf (aref cr-arr0 i) cr (aref ci-arr0 i) ci
                               (aref zr-arr0 i) cr (aref zi-arr0 i) ci
                               (aref start-steps0 i) global-step (aref snap-valid0 i) 0)))
                      ((>= steps buddha::+max-bailout+)
                       (multiple-value-bind (cr ci) (buddha::generate-candidate-f32 rng)
                         (setf (aref cr-arr0 i) cr (aref ci-arr0 i) ci
                               (aref zr-arr0 i) cr (aref zi-arr0 i) ci
                               (aref start-steps0 i) global-step (aref snap-valid0 i) 0))))))

                ;; Vector 1
                (dotimes (i 16)
                  (let ((steps (- global-step (aref start-steps1 i))))
                    (declare (type (unsigned-byte 32) steps))
                    (cond
                      ((logbitp i m1)
                       (when (<= buddha::+min-bailout+ steps buddha::+max-bailout+)
                         (if atomic
                             (buddha::trace-symmetric-orbit-f32-atomic (aref cr-arr1 i) (aref ci-arr1 i) steps
                                                                       array w h px-mul x-off py-mul y-off)
                             (buddha::trace-symmetric-orbit-f32 (aref cr-arr1 i) (aref ci-arr1 i) steps
                                                                array w h px-mul x-off py-mul y-off))
                         (incf hits))
                       (multiple-value-bind (cr ci) (buddha::generate-candidate-f32 rng)
                         (setf (aref cr-arr1 i) cr (aref ci-arr1 i) ci
                               (aref zr-arr1 i) cr (aref zi-arr1 i) ci
                               (aref start-steps1 i) global-step (aref snap-valid1 i) 0)))
                      ((and cycle-detection check-bailout (= (aref snap-valid1 i) 1)
                            (let ((dzr (- (aref zr-arr1 i) (aref snap-zr1 i)))
                                  (dzi (- (aref zi-arr1 i) (aref snap-zi1 i))))
                              (< (+ (* dzr dzr) (* dzi dzi)) buddha::+cycle-eps-sq-f32+)))
                       (multiple-value-bind (cr ci) (buddha::generate-candidate-f32 rng)
                         (setf (aref cr-arr1 i) cr (aref ci-arr1 i) ci
                               (aref zr-arr1 i) cr (aref zi-arr1 i) ci
                               (aref start-steps1 i) global-step (aref snap-valid1 i) 0)))
                      ((>= steps buddha::+max-bailout+)
                       (multiple-value-bind (cr ci) (buddha::generate-candidate-f32 rng)
                         (setf (aref cr-arr1 i) cr (aref ci-arr1 i) ci
                               (aref zr-arr1 i) cr (aref zi-arr1 i) ci
                               (aref start-steps1 i) global-step (aref snap-valid1 i) 0))))))

                ;; Vector 2
                (dotimes (i 16)
                  (let ((steps (- global-step (aref start-steps2 i))))
                    (declare (type (unsigned-byte 32) steps))
                    (cond
                      ((logbitp i m2)
                       (when (<= buddha::+min-bailout+ steps buddha::+max-bailout+)
                         (if atomic
                             (buddha::trace-symmetric-orbit-f32-atomic (aref cr-arr2 i) (aref ci-arr2 i) steps
                                                                       array w h px-mul x-off py-mul y-off)
                             (buddha::trace-symmetric-orbit-f32 (aref cr-arr2 i) (aref ci-arr2 i) steps
                                                                array w h px-mul x-off py-mul y-off))
                         (incf hits))
                       (multiple-value-bind (cr ci) (buddha::generate-candidate-f32 rng)
                         (setf (aref cr-arr2 i) cr (aref ci-arr2 i) ci
                               (aref zr-arr2 i) cr (aref zi-arr2 i) ci
                               (aref start-steps2 i) global-step (aref snap-valid2 i) 0)))
                      ((and cycle-detection check-bailout (= (aref snap-valid2 i) 1)
                            (let ((dzr (- (aref zr-arr2 i) (aref snap-zr2 i)))
                                  (dzi (- (aref zi-arr2 i) (aref snap-zi2 i))))
                              (< (+ (* dzr dzr) (* dzi dzi)) buddha::+cycle-eps-sq-f32+)))
                       (multiple-value-bind (cr ci) (buddha::generate-candidate-f32 rng)
                         (setf (aref cr-arr2 i) cr (aref ci-arr2 i) ci
                               (aref zr-arr2 i) cr (aref zi-arr2 i) ci
                               (aref start-steps2 i) global-step (aref snap-valid2 i) 0)))
                      ((>= steps buddha::+max-bailout+)
                       (multiple-value-bind (cr ci) (buddha::generate-candidate-f32 rng)
                         (setf (aref cr-arr2 i) cr (aref ci-arr2 i) ci
                               (aref zr-arr2 i) cr (aref zi-arr2 i) ci
                               (aref start-steps2 i) global-step (aref snap-valid2 i) 0))))))

                (when (and cycle-detection check-bailout)
                  (dotimes (i 16)
                    (setf (aref snap-zr0 i) (aref zr-arr0 i) (aref snap-zi0 i) (aref zi-arr0 i) (aref snap-valid0 i) 1
                          (aref snap-zr1 i) (aref zr-arr1 i) (aref snap-zi1 i) (aref zi-arr1 i) (aref snap-valid1 i) 1
                          (aref snap-zr2 i) (aref zr-arr2 i) (aref snap-zi2 i) (aref zi-arr2 i) (aref snap-valid2 i) 1)))

                (setf cr-vec0 (sb-simd-avx512f:f32.16-aref cr-arr0 0)
                      ci-vec0 (sb-simd-avx512f:f32.16-aref ci-arr0 0)
                      zr-vec0 (sb-simd-avx512f:f32.16-aref zr-arr0 0)
                      zi-vec0 (sb-simd-avx512f:f32.16-aref zi-arr0 0)
                      cr-vec1 (sb-simd-avx512f:f32.16-aref cr-arr1 0)
                      ci-vec1 (sb-simd-avx512f:f32.16-aref ci-arr1 0)
                      zr-vec1 (sb-simd-avx512f:f32.16-aref zr-arr1 0)
                      zi-vec1 (sb-simd-avx512f:f32.16-aref zi-arr1 0)
                      cr-vec2 (sb-simd-avx512f:f32.16-aref cr-arr2 0)
                      ci-vec2 (sb-simd-avx512f:f32.16-aref ci-arr2 0)
                      zr-vec2 (sb-simd-avx512f:f32.16-aref zr-arr2 0)
                      zi-vec2 (sb-simd-avx512f:f32.16-aref zi-arr2 0)))))))))

;;; ============================================================================
;;; Part 3: Test Runner
;;; ============================================================================

(defun run-register-benchmark ()
  (format t "~%========================================================================~%")
  (format t "TEST 1: Raw Mandelbrot Math & Register Saturation (50,000,000 steps)~%")
  (format t "========================================================================~%")
  (format t "| Implementation | Active Registers | Execution Time | Throughput | vs 1x |~%")
  (format t "|:---|:---|:---|:---|:---|~%")
  (finish-output)

  (let ((n 50000000)
        (base-th 1.0d0))
    ;; 1x
    (let ((t0 (get-internal-real-time)))
      (bench-raw-1x n)
      (let* ((t1 (get-internal-real-time))
             (sec (/ (float (- t1 t0) 1d0) (float internal-time-units-per-second 1d0)))
             (th (/ (* n 16d0) sec 1e9)))
        (setf base-th th)
        (format t "| 1x Vector (16 lanes) | ~2D ZMM regs | ~,3F s | ~,2F G iters/s | 1.00x |~%" 10 sec th)
        (finish-output)))
    ;; 2x
    (let ((t0 (get-internal-real-time)))
      (bench-raw-2x n)
      (let* ((t1 (get-internal-real-time))
             (sec (/ (float (- t1 t0) 1d0) (float internal-time-units-per-second 1d0)))
             (th (/ (* n 32d0) sec 1e9)))
        (format t "| 2x Vector (32 lanes) | ~2D ZMM regs | ~,3F s | ~,2F G iters/s | ~,2Fx |~%"
                19 sec th (/ th base-th))
        (finish-output)))
    ;; 3x
    (let ((t0 (get-internal-real-time)))
      (bench-raw-3x n)
      (let* ((t1 (get-internal-real-time))
             (sec (/ (float (- t1 t0) 1d0) (float internal-time-units-per-second 1d0)))
             (th (/ (* n 48d0) sec 1e9)))
        (format t "| 3x Vector (48 lanes) | ~2D ZMM regs | ~,3F s | ~,2F G iters/s | ~,2Fx |~%"
                26 sec th (/ th base-th))
        (finish-output))))

  (format t "~%========================================================================~%")
  (format t "TEST 2: Real Buddhabrot Worker (Single-Thread, 25,000 Escaped Orbits)~%")
  (format t "========================================================================~%")
  (format t "| Implementation | Active Registers | Worker Time | Orbit Rate | vs 1x |~%")
  (format t "|:---|:---|:---|:---|:---|~%")
  (finish-output)

  (let ((target 25000)
        (arr (make-array (* 12944 8000 3) :element-type '(unsigned-byte 32) :initial-element 0))
        (base-rate 1.0d0))
    ;; 1x
    (sb-ext:gc :full t)
    (fill arr 0)
    (let ((rng (buddha::make-fast-rng 987654321)))
      (let ((t0 (get-internal-real-time)))
        (buddha::simd-worker-f32 target rng arr 12944 8000 nil t)
        (let* ((t1 (get-internal-real-time))
               (sec (/ (float (- t1 t0) 1d0) (float internal-time-units-per-second 1d0)))
               (kops (/ (float target 1d0) sec 1000.0)))
          (setf base-rate kops)
          (format t "| 1x Vector (16 lanes) | ~2D ZMM regs | ~,3F s | ~,1F k orb/s | 1.00x |~%" 10 sec kops)
          (finish-output))))
    ;; 2x
    (sb-ext:gc :full t)
    (fill arr 0)
    (let ((rng (buddha::make-fast-rng 987654321)))
      (let ((t0 (get-internal-real-time)))
        (simd-worker-f32-2x target rng arr 12944 8000 nil t)
        (let* ((t1 (get-internal-real-time))
               (sec (/ (float (- t1 t0) 1d0) (float internal-time-units-per-second 1d0)))
               (kops (/ (float target 1d0) sec 1000.0)))
          (format t "| 2x Vector (32 lanes) | ~2D ZMM regs | ~,3F s | ~,1F k orb/s | ~,2Fx |~%"
                  19 sec kops (/ kops base-rate))
          (finish-output))))
    ;; 3x
    (sb-ext:gc :full t)
    (fill arr 0)
    (let ((rng (buddha::make-fast-rng 987654321)))
      (let ((t0 (get-internal-real-time)))
        (simd-worker-f32-3x target rng arr 12944 8000 nil t)
        (let* ((t1 (get-internal-real-time))
               (sec (/ (float (- t1 t0) 1d0) (float internal-time-units-per-second 1d0)))
               (kops (/ (float target 1d0) sec 1000.0)))
          (format t "| 3x Vector (48 lanes) | ~2D ZMM regs | ~,3F s | ~,1F k orb/s | ~,2Fx |~%"
                  26 sec kops (/ kops base-rate))
          (finish-output)))))

  (format t "~%========================================================================~%")
  (format t "TEST 3: Multi-Threaded Real Buddhabrot (16 Threads, 80,000 Escaped Orbits)~%")
  (format t "========================================================================~%")
  (format t "| Implementation | Active Regs/Th | 16-Thread Time | Throughput | vs 1x |~%")
  (format t "|:---|:---|:---|:---|:---|~%")
  (finish-output)

  (let ((th 16)
        (samples-per-th 5000)
        (arr (make-array (* 12944 8000 3) :element-type '(unsigned-byte 32) :initial-element 0))
        (base-time 1.0d0))
    ;; 1x
    (sb-ext:gc :full t)
    (fill arr 0)
    (let ((t0 (get-internal-real-time)))
      (let ((workers
              (loop for tid below th
                    collect
                    (let ((rng (buddha::make-fast-rng (+ (* (get-internal-real-time) 1000003) (* tid 982451653)))))
                      (sb-thread:make-thread
                       (lambda ()
                         (buddha::simd-worker-f32 samples-per-th rng arr 12944 8000 nil t))
                       :name (format nil "worker-1x-~D" tid))))))
        (mapc #'sb-thread:join-thread workers))
      (let* ((t1 (get-internal-real-time))
             (sec (/ (float (- t1 t0) 1d0) (float internal-time-units-per-second 1d0)))
             (mops (/ (* th samples-per-th 2d0) sec 1e6)))
        (setf base-time sec)
        (format t "| 1x Vector (16 lanes) | ~2D ZMM regs | ~,3F s | ~,1F M orb/s | 1.00x |~%" 10 sec mops)
        (finish-output)))
    ;; 2x
    (sb-ext:gc :full t)
    (fill arr 0)
    (let ((t0 (get-internal-real-time)))
      (let ((workers
              (loop for tid below th
                    collect
                    (let ((rng (buddha::make-fast-rng (+ (* (get-internal-real-time) 1000003) (* tid 982451653)))))
                      (sb-thread:make-thread
                       (lambda ()
                         (simd-worker-f32-2x samples-per-th rng arr 12944 8000 nil t))
                       :name (format nil "worker-2x-~D" tid))))))
        (mapc #'sb-thread:join-thread workers))
      (let* ((t1 (get-internal-real-time))
             (sec (/ (float (- t1 t0) 1d0) (float internal-time-units-per-second 1d0)))
             (mops (/ (* th samples-per-th 2d0) sec 1e6)))
        (format t "| 2x Vector (32 lanes) | ~2D ZMM regs | ~,3F s | ~,1F M orb/s | ~,2Fx |~%"
                19 sec mops (/ base-time sec))
        (finish-output)))
    ;; 3x
    (sb-ext:gc :full t)
    (fill arr 0)
    (let ((t0 (get-internal-real-time)))
      (let ((workers
              (loop for tid below th
                    collect
                    (let ((rng (buddha::make-fast-rng (+ (* (get-internal-real-time) 1000003) (* tid 982451653)))))
                      (sb-thread:make-thread
                       (lambda ()
                         (simd-worker-f32-3x samples-per-th rng arr 12944 8000 nil t))
                       :name (format nil "worker-3x-~D" tid))))))
        (mapc #'sb-thread:join-thread workers))
      (let* ((t1 (get-internal-real-time))
             (sec (/ (float (- t1 t0) 1d0) (float internal-time-units-per-second 1d0)))
             (mops (/ (* th samples-per-th 2d0) sec 1e6)))
        (format t "| 3x Vector (48 lanes) | ~2D ZMM regs | ~,3F s | ~,1F M orb/s | ~,2Fx |~%"
                26 sec mops (/ base-time sec))
        (finish-output)))))

(run-register-benchmark)
(sb-ext:exit)
