;;;; bench-threads.lisp — Thread Scaling Benchmark for Buddhabrot AVX-512
;;;; Evaluates scaling across 4, 8, 10, 12, 16, 20 worker threads.

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

(defun run-thread-benchmark (&key (target-total-orbits 80000) (width 12944) (height 8000))
  (format t "~%========================================================================~%")
  (format t "Buddhabrot AVX-512 Thread Scaling Benchmark~%")
  (format t "Canvas: ~:D x ~:D (~,2F MP), Target Escaped Orbits: ~:D~%"
          width height (/ (* width height) 1e6) target-total-orbits)
  (format t "========================================================================~%~%")
  (format t "| Threads | Samples/Th | Compute Time | Orbit Throughput | EXR Export | Total Time |~%")
  (format t "|:---|:---|:---|:---|:---|:---|~%")
  (finish-output)

  (dolist (th '(4 8 10 12 16 20))
    (let* ((samples-per-th (floor target-total-orbits th))
           (total-target (* th samples-per-th))
           (effective-orbits (* total-target 2))
           (arr (make-array (* width height 3) :element-type '(unsigned-byte 32) :initial-element 0)))
      (sb-ext:gc :full t)
      ;; 1. Compute Phase
      (let ((t0 (get-internal-real-time)))
        (let ((workers
                (loop for tid below th
                      collect
                      (let ((rng (buddha::make-fast-rng (+ (* (get-internal-real-time) 1000003) (* tid 982451653)))))
                        (sb-thread:make-thread
                         (lambda ()
                           (buddha::simd-worker-f32 samples-per-th rng arr width height nil t))
                         :name (format nil "worker-~D" tid))))))
          (mapc #'sb-thread:join-thread workers))
        (let* ((t1 (get-internal-real-time))
               (compute-sec (/ (float (- t1 t0) 1d0) (float internal-time-units-per-second 1d0)))
               (throughput (/ (float effective-orbits 1d0) (max 0.0001d0 compute-sec))))
          ;; 2. EXR Phase
          (let ((tmp-file "/tmp/bench-th.exr")
                (t2 (get-internal-real-time)))
            (buddha:export-exr tmp-file arr width height :threads th :compression :rle)
            (let* ((t3 (get-internal-real-time))
                   (export-sec (/ (float (- t3 t2) 1d0) (float internal-time-units-per-second 1d0)))
                   (total-sec (+ compute-sec export-sec)))
              (format t "| ~2D | ~5D | ~,3F s | ~,1F M orb/s | ~,3F s | ~,3F s |~%"
                      th samples-per-th compute-sec (/ throughput 1000000.0) export-sec total-sec)
              (finish-output)
              (when (probe-file tmp-file)
                (delete-file tmp-file)))))))))

(run-thread-benchmark)
(sb-ext:exit)
