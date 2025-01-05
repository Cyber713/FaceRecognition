/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package cg.widgets.face.recognition

import android.app.AlertDialog
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.DialogInterface
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.hardware.camera2.CameraCharacteristics
import android.media.ImageReader.OnImageAvailableListener
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.util.TypedValue
import android.view.View
import android.widget.EditText
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.core.content.ContextCompat
import cg.widgets.face.recognition.activities.CameraActivity
import com.google.android.gms.tasks.OnSuccessListener
import com.google.android.material.floatingactionbutton.FloatingActionButton
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetector
import com.google.mlkit.vision.face.FaceDetectorOptions
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.detection.customview.OverlayView
import org.tensorflow.lite.examples.detection.env.BorderedText
import org.tensorflow.lite.examples.detection.env.ImageUtils
import org.tensorflow.lite.examples.detection.env.Logger
import org.tensorflow.lite.examples.detection.tflite.SimilarityClassifier
import org.tensorflow.lite.examples.detection.tflite.SimilarityClassifier.Recognition
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.Arrays
import java.util.LinkedList

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
class DetectorActivity : CameraActivity(),
    OnImageAvailableListener {
    var trackingOverlay: OverlayView? = null
    private var sensorOrientation: Int? = null

    private var detector: SimilarityClassifier? = null

    private var lastProcessingTimeMs: Long = 0
    private var rgbFrameBitmap: Bitmap? = null
    private var croppedBitmap: Bitmap? = null
    private var cropCopyBitmap: Bitmap? = null

    private var computingDetection = false
    private var addPending = false

    //private boolean adding = false;
    private var timestamp: Long = 0

    private var frameToCropTransform: Matrix? = null
    private var cropToFrameTransform: Matrix? = null

    //private Matrix cropToPortraitTransform;
    private var tracker: MultiBoxTracker? = null

    private var borderedText: BorderedText? = null

    // Face detector
    private var faceDetector: FaceDetector? = null

    // here the preview image is drawn in portrait way
    private var portraitBmp: Bitmap? = null

    // here the face is cropped and drawn
    private var faceBmp: Bitmap? = null

    private lateinit var faceNetModelInterpreter: Interpreter

    private var fabAdd: FloatingActionButton? = null
    private lateinit var faceNetImageProcessor: ImageProcessor

    fun initializeInterpreter(assetManager: AssetManager, modelPath: String) {
        val model = loadModelFile(assetManager, modelPath)
        val options = Interpreter.Options().apply {
            setNumThreads(4)  // Optimize for performance
        }
        faceNetModelInterpreter = Interpreter(model, options)
    }

    private fun loadModelFile(assetManager: AssetManager, modelPath: String): MappedByteBuffer {
        val fileDescriptor = assetManager.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    //private HashMap<String, Classifier.Recognition> knownFaces = new HashMap<>();
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        initializeInterpreter(this.assets, "mobile_face_net.tflite")
        faceNetImageProcessor = ImageProcessor.Builder()
            .add(
                ResizeOp(
                    112,
                    112,
                    ResizeOp.ResizeMethod.BILINEAR
                )
            )
            .add(NormalizeOp(0f, 255f))
            .build()
        fabAdd = findViewById(R.id.fab_add)



        fabAdd!!.setOnClickListener(View.OnClickListener { onAddClick() })

        // Real-time contour detection of multiple faces
        val options =
            FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                .setContourMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                .build()


        val detector = FaceDetection.getClient(options)

        faceDetector = detector

detector.process(drawableToBitmap(this,R.drawable.smiley),0).addOnSuccessListener {
    loadRecognitions()
}


//        loadRecognitions();
        //checkWritePermission();

    }

    fun drawableToBitmap(context: Context, drawableId: Int): Bitmap {
        val drawable = ContextCompat.getDrawable(context, drawableId)
        val bitmap = Bitmap.createBitmap(
            drawable!!.intrinsicWidth,
            drawable.intrinsicHeight,
            Bitmap.Config.ARGB_8888
        )
        val canvas = Canvas(bitmap)
        drawable.setBounds(0, 0, canvas.width, canvas.height)
        drawable.draw(canvas)
        return bitmap
    }

    fun reconverter(input: String): Array<FloatArray> {
        return input
            .removeSurrounding("[", "]") // Remove outer brackets
            .split("], [")               // Split into rows
            .map { row ->
                row
                    .removeSurrounding("[", "]")  // Remove inner brackets
                    .split(", ")                  // Split values
                    .map { it.toFloat() }         // Convert to float
                    .toFloatArray()               // Convert to FloatArray
            }
            .toTypedArray() // Convert to Array<FloatArray>
    }

    private fun loadRecognitions() {
        val rec = SimilarityClassifier.Recognition(
            "0", "Hello", -1.0f,
            RectF(156.0f, 160.0f, 340.0f, 342.0f)
        )
        val tempVector = "[[0.002438068, 0.009879535, 0.0021977718, -0.01259324, -0.018551938, -0.024603445, -0.013274414, 0.070915766, -0.071284294, -0.006205039, -0.011144604, 0.0027010515, -0.0062769265, 0.0070215636, -3.20317E-4, -0.032882173, 4.3491428E-4, -0.0015151236, 5.703661E-4, 0.0044415467, -0.016834078, 0.013155096, 0.020414306, 0.0063589667, 0.18544433, 0.007808325, -3.577042E-4, -0.0026789466, 0.026046285, -0.19970258, -0.0031413718, -0.012094521, 0.30632848, 0.006183326, -5.4811075E-4, -0.14765282, -0.12472348, -0.025774185, 0.009551149, 0.2417348, -0.003695652, 0.0018943588, -0.0054013995, -0.011387395, 0.014985515, 0.04878188, -0.1406221, 0.08659876, -0.005426187, -0.034122825, -0.025004817, -1.6975141E-4, -0.03552976, -0.004594279, 0.118358135, 1.654705E-4, 0.038115256, -0.0026995335, -0.036914278, 0.014589321, 0.019840129, -0.048870463, 0.12419196, 0.23463817, 0.0014854588, 0.11835555, -0.002382819, -0.027057115, 0.0063888174, 9.7151083E-4, -0.010002272, -0.02285439, -0.13973777, 8.618351E-4, 0.0816185, 0.017611235, 0.0021615329, -0.0042456817, -0.0042732926, -0.13110466, 0.008759852, -0.021468988, -0.0052682464, 0.12954208, 0.094278306, 0.0025400638, -2.1155862E-4, 0.01693039, -0.08631775, -0.106425375, 0.059185933, -0.006021146, -0.006075135, -0.003908577, -0.21651104, -0.072077654, -0.13718683, 0.12252652, -0.004675431, 0.0064243646, 0.0016767525, 0.006299671, -0.006266839, 9.0418145E-4, -0.007958147, 0.0065695737, 0.053450678, 0.007009268, 0.019554147, -0.0010038788, 0.036879662, 0.006664892, -0.01713726, -0.06417755, 0.0018804405, -0.0127574615, -0.005053232, -0.003922114, -0.0342852, -0.080743834, 0.0073289196, -0.00653957, -0.27276394, -0.0015694731, 0.0029046328, 0.008297218, -0.002467507, -0.001726068, 4.5200507E-4, -0.11235935, -0.0026853278, -0.0063364557, 4.6136858E-5, -0.074399136, 0.010260809, -0.0069119963, -0.020683598, -0.0028673497, -0.0055735777, -1.3097365E-4, 0.0030213397, 0.0034276133, 0.009680732, 0.18259212, 0.030702207, -0.02344819, 0.001528792, -0.0018901551, 0.0057934253, 0.010463758, 0.0017481905, 0.047172446, 0.26719716, 7.6597475E-4, 0.006029361, 0.0108897565, -0.0075974087, 0.027222464, -0.09821197, 3.7264146E-4, -0.018066458, -0.0016640484, 0.0022101575, -0.002206966, 0.002925807, 0.005108162, -0.0068637584, 0.24388537, 0.0043733157, -0.0067722145, 0.28231505, 0.0015768601, 0.0055360077, 0.0142879, -0.008716619, 0.0011579667, -0.026725609, -0.011923737, -4.1007603E-4, -0.007170402, -0.03889005, 0.07202625, -0.0034067207, 0.0026672585, 0.06563352, -0.034255765, -0.0026640787, -0.002769291, 0.028540574, -0.13810244, -0.011414741, -0.010171691]]"


        rec.extra = reconverter(tempVector)
        rec.color = Color.RED
        if (detector != null) {
            detector!!.register("John", rec)
        }else{
            Handler(Looper.getMainLooper()).postDelayed({ detector!!.register("Worked", rec)
            }, 100)


        }
    }


    private fun onAddClick() {
        addPending = true
    }

    public override fun onPreviewSizeChosen(size: Size, rotation: Int) {
        val textSizePx =
            TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, resources.displayMetrics
            )
        borderedText = BorderedText(textSizePx)
        borderedText!!.setTypeface(Typeface.MONOSPACE)

        tracker = MultiBoxTracker(this)


        try {
            detector =
                TFLiteObjectDetectionAPIModel.create(
                    assets,
                    TF_OD_API_MODEL_FILE,
                    TF_OD_API_LABELS_FILE,
                    TF_OD_API_INPUT_SIZE,
                    TF_OD_API_IS_QUANTIZED
                )
            //cropSize = TF_OD_API_INPUT_SIZE;
        } catch (e: IOException) {
            e.printStackTrace()
            LOGGER.e(e, "Exception initializing classifier!")
            val toast =
                Toast.makeText(
                    applicationContext, "Classifier could not be initialized", Toast.LENGTH_SHORT
                )
            toast.show()
            finish()
        }

        previewWidth = size.width
        previewHeight = size.height

        sensorOrientation = rotation - screenOrientation
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation)

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight)
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Bitmap.Config.ARGB_8888)


        val targetW: Int
        val targetH: Int
        if (sensorOrientation == 90 || sensorOrientation == 270) {
            targetH = previewWidth
            targetW = previewHeight
        } else {
            targetW = previewWidth
            targetH = previewHeight
        }
        val cropW = (targetW / 2.0).toInt()
        val cropH = (targetH / 2.0).toInt()

        croppedBitmap = Bitmap.createBitmap(cropW, cropH, Bitmap.Config.ARGB_8888)

        portraitBmp = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
        faceBmp =
            Bitmap.createBitmap(TF_OD_API_INPUT_SIZE, TF_OD_API_INPUT_SIZE, Bitmap.Config.ARGB_8888)

        frameToCropTransform =
            ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight,
                cropW, cropH,
                sensorOrientation!!, MAINTAIN_ASPECT
            )

        //    frameToCropTransform =
//            ImageUtils.getTransformationMatrix(
//                    previewWidth, previewHeight,
//                    previewWidth, previewHeight,
//                    sensorOrientation, MAINTAIN_ASPECT);
        cropToFrameTransform = Matrix()
        frameToCropTransform!!.invert(cropToFrameTransform)


        val frameToPortraitTransform =
            ImageUtils.getTransformationMatrix(
                previewWidth, previewHeight,
                targetW, targetH,
                sensorOrientation!!, MAINTAIN_ASPECT
            )



        trackingOverlay = findViewById<View>(R.id.tracking_overlay) as OverlayView
        trackingOverlay!!.addCallback { canvas ->
            tracker!!.draw(canvas)
            if (isDebug) {
                tracker!!.drawDebug(canvas)
            }
        }

        tracker!!.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation!!)
    }


    override fun processImage() {
        ++timestamp
        val currTimestamp = timestamp
        trackingOverlay!!.postInvalidate()

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage()
            return
        }
        computingDetection = true

        LOGGER.i("Preparing image $currTimestamp for detection in bg thread.")

        rgbFrameBitmap!!.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight)

        readyForNextImage()

        val canvas = Canvas(croppedBitmap!!)
        canvas.drawBitmap(rgbFrameBitmap!!, frameToCropTransform!!, null)
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap)
        }

        val image = InputImage.fromBitmap(croppedBitmap!!, 0)
        faceDetector!!.process(image)
            .addOnSuccessListener(OnSuccessListener<List<Face>> { faces ->
                if (faces.size == 0) {
                    updateResults(currTimestamp, LinkedList())
                    return@OnSuccessListener
                }
                runInBackground {
                    onFacesDetected(currTimestamp, faces, addPending)
                    addPending = false
                }
            })
    }

    override fun getLayoutId(): Int {
        return R.layout.tfe_od_camera_connection_fragment_tracking
    }

    override fun getDesiredPreviewFrameSize(): Size {
        return DESIRED_PREVIEW_SIZE
    }

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum class DetectorMode {
        TF_OD_API
    }

    override fun setUseNNAPI(isChecked: Boolean) {
        runInBackground { detector!!.setUseNNAPI(isChecked) }
    }

    override fun setNumThreads(numThreads: Int) {
        runInBackground { detector!!.setNumThreads(numThreads) }
    }


    // Face Processing
    private fun createTransform(
        srcWidth: Int,
        srcHeight: Int,
        dstWidth: Int,
        dstHeight: Int,
        applyRotation: Int
    ): Matrix {
        val matrix = Matrix()
        if (applyRotation != 0) {
            if (applyRotation % 90 != 0) {
                LOGGER.w("Rotation of %d % 90 != 0", applyRotation)
            }

            // Translate so center of image is at origin.
            matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f)

            // Rotate around origin.
            matrix.postRotate(applyRotation.toFloat())
        }

        //        // Account for the already applied rotation, if any, and then determine how
//        // much scaling is needed for each axis.
//        final boolean transpose = (Math.abs(applyRotation) + 90) % 180 == 0;
//        final int inWidth = transpose ? srcHeight : srcWidth;
//        final int inHeight = transpose ? srcWidth : srcHeight;
        if (applyRotation != 0) {
            // Translate back from origin centered reference to destination frame.

            matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f)
        }

        return matrix
    }

    private fun showAddFaceDialog(rec: Recognition) {
        val builder = AlertDialog.Builder(this)
        val inflater = layoutInflater
        val dialogLayout = inflater.inflate(R.layout.image_edit_dialog, null)
        val ivFace = dialogLayout.findViewById<ImageView>(R.id.dlg_image)
        val tvTitle = dialogLayout.findViewById<TextView>(R.id.dlg_title)
        val etName = dialogLayout.findViewById<EditText>(R.id.dlg_input)

        tvTitle.text = "Add Face"
        ivFace.setImageBitmap(rec.crop)
        etName.hint = "Input name"

        builder.setPositiveButton("OK", DialogInterface.OnClickListener { dlg, i ->
            val name = etName.text.toString()
            if (name.isEmpty()) {
                return@OnClickListener
            }
            detector!!.register(name, rec)

            val tensorImage = TensorImage.fromBitmap(rec.crop)

            val faceNetByteBuffer: ByteBuffer = faceNetImageProcessor.process(tensorImage).buffer
            val faceOutputArray = Array(1) { FloatArray(192) }
            faceNetModelInterpreter.run(faceNetByteBuffer, faceOutputArray)

            if (true) {
                val readableString = (rec.extra as Array<FloatArray>).joinToString(
                    prefix = "[",
                    postfix = "]"
                ) { it.joinToString(", ", "[", "]") }

                copyToClipboard("Extra: " + readableString)

                Log.i("Distance", rec.distance.toString())
                Log.i("L_L", rec.location.left.toString())
                Log.i("L_R", rec.location.right.toString())
                Log.i("L_B", rec.location.bottom.toString())
                Log.i("L_T", rec.location.top.toString())

                Log.i("Extra", readableString);


                Toast.makeText(this, "Did", Toast.LENGTH_SHORT).show()
            } else {
                println("Object is not a 2D float array.")
            }

//            try {
//                copyToClipboard(
//                    "Result: "+Arrays.deepToString(faceOutputArray)
//                )            }catch (_: Exception){
//
//            }

//            detector!!.register("Hello",Recognition("","",-1.0f,RectF(138.0f,170.0f,382.0f,410.0f)))

//            copyToClipboard(
//                "" +
//                        "id: ${rec.id} "+
//                        "title: ${rec.title} "+
//                        "distance: ${rec.distance} "+
//                        "location: ${rec.location} "
//            )
//            copyToClipboard(rec.toString())
//            saveRecognitionData(name,rec)
            //knownFaces.put(name, rec);
            dlg.dismiss()
        })
        builder.setView(dialogLayout)
        builder.show()
    }


    private fun saveRecognitionData(name: String, rec: Recognition) {
        // Create a directory to store face images
        val directory = File(getExternalFilesDir(null), "recognized_faces")
        if (!directory.exists()) {
            directory.mkdirs()
        }

        // Save the cropped face image to the directory
        val imageFile = File(directory, "$name.jpg")
        copyToClipboard(imageFile.path)
        try {
            val outputStream = FileOutputStream(imageFile)
            rec.crop.compress(Bitmap.CompressFormat.JPEG, 100, outputStream)
            outputStream.close()

            // You can save metadata as well, such as name and additional info
            val metadata = File(directory, "metadata.txt")
            val metadataContent = "Name: $name\nAdditional Info: ${rec.extra.toString()}"
            metadata.writeText(metadataContent)

            // Optionally show a success message
            Toast.makeText(this, "Face saved successfully!", Toast.LENGTH_SHORT).show()
        } catch (e: IOException) {
            e.printStackTrace()
            Toast.makeText(this, "Error saving face data", Toast.LENGTH_SHORT).show()
        }
    }
//
//    fun loadRecognitionData(): List<File> {
//        val directory = File(filesDir, "recognized_faces")
//        val files = directory.listFiles()
//        return files?.toList() ?: emptyList()
//    }


    private fun updateResults(currTimestamp: Long, mappedRecognitions: List<Recognition>) {
        tracker!!.trackResults(mappedRecognitions, currTimestamp)
        trackingOverlay!!.postInvalidate()
        computingDetection = false


        //adding = false;
        if (mappedRecognitions.size > 0) {
            LOGGER.i("Adding results")
            val rec = mappedRecognitions[0]
            if (rec.extra != null) {
                showAddFaceDialog(rec)
            }
        }

        runOnUiThread {
            showFrameInfo(previewWidth.toString() + "x" + previewHeight)
            showCropInfo(croppedBitmap!!.width.toString() + "x" + croppedBitmap!!.height)
            showInference(lastProcessingTimeMs.toString() + "ms")
        }
    }

    private fun onFacesDetected(currTimestamp: Long, faces: List<Face>, add: Boolean) {
        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap!!)
        val canvas = Canvas(cropCopyBitmap!!)
        val paint = Paint()
        paint.color = Color.RED
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 2.0f
        val tensorImage = TensorImage.fromBitmap(cropCopyBitmap)

        val faceNetByteBuffer: ByteBuffer = faceNetImageProcessor.process(tensorImage).buffer
        val faceOutputArray = Array(1) { FloatArray(192) }
        faceNetModelInterpreter.run(faceNetByteBuffer, faceOutputArray)

        try {
//            copyToClipboard(faceOutputArray.toString())
        } catch (_: Exception) {

        }
        var minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API
        when (MODE) {
            DetectorMode.TF_OD_API -> minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API
        }

        val mappedRecognitions: MutableList<Recognition> =
            LinkedList()


        //final List<Classifier.Recognition> results = new ArrayList<>();

        // Note this can be done only once
        val sourceW = rgbFrameBitmap!!.width
        val sourceH = rgbFrameBitmap!!.height
        val targetW = portraitBmp!!.width
        val targetH = portraitBmp!!.height
        val transform = createTransform(
            sourceW,
            sourceH,
            targetW,
            targetH,
            sensorOrientation!!
        )
        val cv = Canvas(portraitBmp!!)

        // draws the original image in portrait mode.
        cv.drawBitmap(rgbFrameBitmap!!, transform, null)

        val cvFace = Canvas(faceBmp!!)

        val saved = false

        for (face in faces) {
            LOGGER.i("FACE$face")
            LOGGER.i("Running detection on face $currTimestamp")

            //results = detector.recognizeImage(croppedBitmap);
            val boundingBox = RectF(face.boundingBox)

            //final boolean goodConfidence = result.getConfidence() >= minimumConfidence;
            val goodConfidence = true //face.get;
            if (boundingBox != null && goodConfidence) {
                // maps crop coordinates to original

                cropToFrameTransform!!.mapRect(boundingBox)

                // maps original coordinates to portrait coordinates
                val faceBB = RectF(boundingBox)
                transform.mapRect(faceBB)

                // translates portrait to origin and scales to fit input inference size
                //cv.drawRect(faceBB, paint);
                val sx = (TF_OD_API_INPUT_SIZE.toFloat()) / faceBB.width()
                val sy = (TF_OD_API_INPUT_SIZE.toFloat()) / faceBB.height()
                val matrix = Matrix()
                matrix.postTranslate(-faceBB.left, -faceBB.top)
                matrix.postScale(sx, sy)

                cvFace.drawBitmap(portraitBmp!!, matrix, null)

                //canvas.drawRect(faceBB, paint);
                var label = ""
                var confidence = -1f
                var color = Color.BLUE
                var extra: Any? = null
                var crop: Bitmap? = null

                if (add) {
                    crop = Bitmap.createBitmap(
                        portraitBmp!!,
                        faceBB.left.toInt(),
                        faceBB.top.toInt(),
                        faceBB.width().toInt(),
                        faceBB.height().toInt()
                    )
//                    copyToClipboard(matrix.toString())
                }

                val startTime = SystemClock.uptimeMillis()
                val resultsAux = detector!!.recognizeImage(faceBmp, add)
                lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime

                if (resultsAux.size > 0) {
                    val result = resultsAux[0]

                    extra = result.extra

//                   copyToClipboard(resultsAux.toString())
                    //          Object extra = result.getExtra();
//          if (extra != null) {
//            LOGGER.i("embeeding retrieved " + extra.toString());
//          }
                    val conf = result.distance
                    if (conf < 1.0f) {
                        confidence = conf
                        label = result.title
                        color = if (result.id == "0") {
                            Color.GREEN
                        } else {
                            Color.RED
                        }
                    }
                }

                if (cameraFacing == CameraCharacteristics.LENS_FACING_FRONT) {
                    // camera is frontal so the image is flipped horizontally
                    // flips horizontally

                    val flip = Matrix()
                    if (sensorOrientation == 90 || sensorOrientation == 270) {
                        flip.postScale(1f, -1f, previewWidth / 2.0f, previewHeight / 2.0f)
                    } else {
                        flip.postScale(-1f, 1f, previewWidth / 2.0f, previewHeight / 2.0f)
                    }
                    //flip.postScale(1, -1, targetW / 2.0f, targetH / 2.0f);
                    flip.mapRect(boundingBox)
                }

                val result = Recognition(
                    "0", label, confidence, boundingBox
                )

                result.color = color
                result.location = boundingBox
                result.extra = extra
                result.crop = crop
                mappedRecognitions.add(result)



                try {
                    Log.e("Extra: ", result.extra.toString())
//                    copyToClipboard(result.extra.toString())

                } catch (_: Exception) {
//                    copyToClipboard("hel")
                }
            }
        }

        //    if (saved) {
//      lastSaved = System.currentTimeMillis();
//    }
        updateResults(currTimestamp, mappedRecognitions)
    }


    companion object {
        private val LOGGER = Logger()


        // FaceNet
        //  private static final int TF_OD_API_INPUT_SIZE = 160;
        //  private static final boolean TF_OD_API_IS_QUANTIZED = false;
        //  private static final String TF_OD_API_MODEL_FILE = "facenet.tflite";
        //  //private static final String TF_OD_API_MODEL_FILE = "facenet_hiroki.tflite";
        // MobileFaceNet
        private const val TF_OD_API_INPUT_SIZE = 112
        private const val TF_OD_API_IS_QUANTIZED = false
        private const val TF_OD_API_MODEL_FILE = "mobile_face_net.tflite"


        private const val TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt"

        private val MODE = DetectorMode.TF_OD_API

        // Minimum detection confidence to track a detection.
        private const val MINIMUM_CONFIDENCE_TF_OD_API = 0.5f
        private const val MAINTAIN_ASPECT = false

        private val DESIRED_PREVIEW_SIZE = Size(640, 480)


        //private static final int CROP_SIZE = 320;
        //private static final Size CROP_SIZE = new Size(320, 320);
        private const val SAVE_PREVIEW_BITMAP = false
        private const val TEXT_SIZE_DIP = 10f
    }

    fun copyToClipboard(text: String?) {
        val clipboard = getSystemService(CLIPBOARD_SERVICE) as ClipboardManager
        val clip = ClipData.newPlainText("Test", text)
        clipboard.setPrimaryClip(clip)
    }
}