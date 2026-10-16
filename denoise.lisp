;;;; denoise.lisp — Standalone Intel Open Image Denoise (OIDN) Interface for SBCL
;;;;
;;;; GPU-accelerated ray tracing image denoiser using SBCL's native sb-alien FFI.
;;;; Supports OIDN 2.x via CUDA, SYCL, HIP, Metal, or CPU.
;;;;
;;;; Copyright (c) 2026 Arthur Miller <arthur.miller@live.com>
;;;; BSD 2-Clause License.

(in-package :cl-user)

(defpackage #:oidn
  (:use #:cl)
  (:export #:*oidn-loaded*
           #:load-oidn
           #:denoise-buffer
           #:denoise-buffer-thread
           #:denoise-buffer-async
           #:device-type-name))

(in-package :oidn)

(declaim (optimize (speed 3) (safety 1) (debug 0)))


;;; Shared Library Loading & Alien Bindings

(defvar *oidn-loaded* nil
  "Tracks whether libOpenImageDenoise.so.2 has been successfully loaded.")

(defun load-oidn (&optional (lib-name "libOpenImageDenoise.so.2"))
  "Load the OIDN shared library dynamically via SB-ALIEN."
  (unless *oidn-loaded*
    (handler-case
        (progn
          (sb-alien:load-shared-object lib-name)
          (setf *oidn-loaded* t))
      (error (e)
        (warn "Could not load ~A: ~A" lib-name e)
        nil))))

;; Auto-load at load/compile time
(eval-when (:compile-toplevel :load-toplevel :execute)
  (handler-case
      (sb-alien:load-shared-object "libOpenImageDenoise.so.2")
    (error () nil)))

(sb-alien:define-alien-routine ("oidnNewDevice" %oidn-new-device) sb-alien:system-area-pointer
  (type sb-alien:int))

(sb-alien:define-alien-routine ("oidnCommitDevice" %oidn-commit-device) sb-alien:void
  (device sb-alien:system-area-pointer))

(sb-alien:define-alien-routine ("oidnGetDeviceInt" %oidn-get-device-int) sb-alien:int
  (device sb-alien:system-area-pointer)
  (name sb-alien:c-string))

(sb-alien:define-alien-routine ("oidnGetDeviceError" %oidn-get-device-error) sb-alien:int
  (device sb-alien:system-area-pointer)
  (out-message (* sb-alien:c-string)))

(sb-alien:define-alien-routine ("oidnNewBuffer" %oidn-new-buffer) sb-alien:system-area-pointer
  (device sb-alien:system-area-pointer)
  (byte-size sb-alien:unsigned-long))

(sb-alien:define-alien-routine ("oidnGetBufferData" %oidn-get-buffer-data) sb-alien:system-area-pointer
  (buffer sb-alien:system-area-pointer))

(sb-alien:define-alien-routine ("oidnWriteBuffer" %oidn-write-buffer) sb-alien:void
  (buffer sb-alien:system-area-pointer)
  (byte-offset sb-alien:unsigned-long)
  (byte-size sb-alien:unsigned-long)
  (src-host-ptr sb-alien:system-area-pointer))

(sb-alien:define-alien-routine ("oidnReadBuffer" %oidn-read-buffer) sb-alien:void
  (buffer sb-alien:system-area-pointer)
  (byte-offset sb-alien:unsigned-long)
  (byte-size sb-alien:unsigned-long)
  (dst-host-ptr sb-alien:system-area-pointer))

(sb-alien:define-alien-routine ("oidnNewFilter" %oidn-new-filter) sb-alien:system-area-pointer
  (device sb-alien:system-area-pointer)
  (type sb-alien:c-string))

(sb-alien:define-alien-routine ("oidnSetFilterImage" %oidn-set-filter-image) sb-alien:void
  (filter sb-alien:system-area-pointer)
  (name sb-alien:c-string)
  (buffer sb-alien:system-area-pointer)
  (format sb-alien:int)
  (width sb-alien:unsigned-long)
  (height sb-alien:unsigned-long)
  (byte-offset sb-alien:unsigned-long)
  (pixel-stride sb-alien:unsigned-long)
  (row-stride sb-alien:unsigned-long))

(sb-alien:define-alien-routine ("oidnSetFilterBool" %oidn-set-filter-bool) sb-alien:void
  (filter sb-alien:system-area-pointer)
  (name sb-alien:c-string)
  (value sb-alien:int))

(sb-alien:define-alien-routine ("oidnCommitFilter" %oidn-commit-filter) sb-alien:void
  (filter sb-alien:system-area-pointer))

(sb-alien:define-alien-routine ("oidnExecuteFilter" %oidn-execute-filter) sb-alien:void
  (filter sb-alien:system-area-pointer))

(sb-alien:define-alien-routine ("oidnReleaseBuffer" %oidn-release-buffer) sb-alien:void
  (buffer sb-alien:system-area-pointer))

(sb-alien:define-alien-routine ("oidnReleaseFilter" %oidn-release-filter) sb-alien:void
  (filter sb-alien:system-area-pointer))

(sb-alien:define-alien-routine ("oidnReleaseDevice" %oidn-release-device) sb-alien:void
  (device sb-alien:system-area-pointer))


;;; Helper Functions

(defun device-type-name (type-code)
  "Convert OIDN integer device type code to human-readable string."
  (case type-code
    (0 "Default Accelerator")
    (1 "CPU")
    (2 "SYCL")
    (3 "CUDA (NVIDIA GPU)")
    (4 "HIP (AMD GPU)")
    (5 "Metal (Apple GPU)")
    (otherwise (format nil "Device(~D)" type-code))))


;;; Core Denoising Pipelinep

(defun denoise-buffer (array width height &key (scale 0.001f0) (threads 16) (hdr t) (device-type 0))
  "Takes 32-bit unsigned hit-count ARRAY (interleaved RGB, length (* width height 3)),
runs Intel Open Image Denoise (OIDN) using the specified DEVICE-TYPE (0 = auto/default GPU),
and returns a new 32-bit unsigned integer array with denoised hit counts."
  (declare (type (simple-array (unsigned-byte 32) (*)) array)
           (type fixnum width height threads device-type)
           (type single-float scale)
           (type boolean hdr))
  (let* ((pixel-count (the fixnum (* width height 3)))
         (byte-size (* pixel-count 4))
         (float-buf (make-array pixel-count :element-type 'single-float))
         (num-workers (max 1 threads))
         (chunk-size (floor pixel-count num-workers))
         (inv-scale (/ 1.0f0 scale)))
    (declare (type fixnum pixel-count byte-size num-workers chunk-size)
             (type single-float inv-scale)
             (type (simple-array single-float (*)) float-buf))

    ;; 1. Multi-threaded integer hit-counts -> float32 conversion
    (let ((workers
            (loop for tid below num-workers
                  collect
                  (let* ((start (* tid chunk-size))
                         (end (if (= tid (1- num-workers)) pixel-count (* (1+ tid) chunk-size))))
                    (sb-thread:make-thread
                     (lambda ()
                       (declare (optimize (speed 3) (safety 0)))
                       (loop for i of-type fixnum from start below end do
                         (setf (aref float-buf i) (* (float (aref array i) 1.0f0) scale))))
                     :name (format nil "oidn-int2flt-~D" tid))))))
      (mapc #'sb-thread:join-thread workers))

    ;; 2. Initialize OIDN device
    (let* ((device (%oidn-new-device device-type))
           (_ (%oidn-commit-device device))
           (dev-type (%oidn-get-device-int device "type"))
           (color-buf (%oidn-new-buffer device byte-size))
           (filter (%oidn-new-filter device "RT")))
      (declare (ignore _))
      (format t "OIDN device: ~A (type=~D)~%" (device-type-name dev-type) dev-type)
      (finish-output)

      ;; 3. Stream float data to OIDN device buffer
      (sb-sys:with-pinned-objects (float-buf)
        (%oidn-write-buffer color-buf 0 byte-size (sb-sys:vector-sap float-buf)))

      ;; 4. Configure filter (OIDN_FORMAT_FLOAT3 = 3)
      (%oidn-set-filter-image filter "color" color-buf 3 width height 0 0 0)
      (%oidn-set-filter-image filter "output" color-buf 3 width height 0 0 0)
      (%oidn-set-filter-bool filter "hdr" (if hdr 1 0))
      (%oidn-commit-filter filter)

      ;; 5. Execute denoiser
      (let ((t0 (get-internal-real-time)))
        (%oidn-execute-filter filter)
        (let* ((t1 (get-internal-real-time))
               (sec (/ (float (- t1 t0) 1d0) (float internal-time-units-per-second 1d0))))
          (format t "OIDN filter executed in ~,3F seconds.~%" sec)
          (finish-output)))

      ;; 6. Read back denoised float data (reusing float-buf)
      (sb-sys:with-pinned-objects (float-buf)
        (%oidn-read-buffer color-buf 0 byte-size (sb-sys:vector-sap float-buf)))

      ;; Clean up OIDN resources promptly
      (%oidn-release-filter filter)
      (%oidn-release-buffer color-buf)
      (%oidn-release-device device))

    ;; 7. Multi-threaded float32 -> 32-bit unsigned integer conversion
    (let ((result-int (make-array pixel-count :element-type '(unsigned-byte 32))))
      (let ((workers
              (loop for tid below num-workers
                    collect
                    (let* ((start (* tid chunk-size))
                           (end (if (= tid (1- num-workers)) pixel-count (* (1+ tid) chunk-size))))
                      (sb-thread:make-thread
                       (lambda ()
                         (declare (optimize (speed 3) (safety 0)))
                         (loop for i of-type fixnum from start below end do
                           (let* ((f (aref float-buf i))
                                  (v (round (* f inv-scale))))
                             (declare (type single-float f) (type fixnum v))
                             (setf (aref result-int i)
                                   (if (< v 0) 0 (if (> v #xFFFFFFFF) #xFFFFFFFF v))))))
                       :name (format nil "oidn-flt2int-~D" tid))))))
        (mapc #'sb-thread:join-thread workers))
      result-int)))


;;; Threaded & Asynchronous Wrappers

(defun denoise-buffer-thread (array width height &key (scale 0.001f0) (threads 16) (hdr t) (device-type 0))
  "Runs DENOISE-BUFFER inside a dedicated worker thread and joins it, ensuring complete
thread and foreign device context isolation."
  (let ((result nil))
    (let ((worker
            (sb-thread:make-thread
             (lambda ()
               (setf result (denoise-buffer array width height
                                            :scale scale :threads threads :hdr hdr :device-type device-type)))
             :name "oidn-worker-thread")))
      (sb-thread:join-thread worker))
    result))

(defun denoise-buffer-async (array width height &key (scale 0.001f0) (threads 16) (hdr t) (device-type 0) callback)
  "Runs DENOISE-BUFFER asynchronously in its own dedicated worker thread.
Returns the worker thread object immediately. If CALLBACK is supplied, it is invoked
with the resulting denoised array upon completion."
  (sb-thread:make-thread
   (lambda ()
     (let ((res (denoise-buffer array width height
                                :scale scale :threads threads :hdr hdr :device-type device-type)))
       (when callback
         (funcall callback res))
       res))
   :name "oidn-async-thread"))
