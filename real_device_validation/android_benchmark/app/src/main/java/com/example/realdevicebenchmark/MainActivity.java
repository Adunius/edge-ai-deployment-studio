package com.example.realdevicebenchmark;

import android.app.Activity;
import android.os.Build;
import android.os.Debug;
import android.os.Bundle;
import android.os.Process;
import android.os.SystemClock;
import android.util.Log;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ScrollView;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;
import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class MainActivity extends Activity {
    private static final String TAG = "RealDeviceBenchmark";
    private static final int WARMUP_RUNS = 10;
    private static final int MEASURED_RUNS = 50;

    private TextView output;
    private Button runButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        ScrollView scrollView = new ScrollView(this);
        output = new TextView(this);
        output.setTextSize(14);
        output.setPadding(32, 32, 32, 32);

        runButton = new Button(this);
        runButton.setText("Run benchmark");
        runButton.setOnClickListener(v -> runBenchmarkAsync());

        android.widget.LinearLayout layout = new android.widget.LinearLayout(this);
        layout.setOrientation(android.widget.LinearLayout.VERTICAL);
        layout.addView(runButton, new ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        ));
        layout.addView(output, new ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
        ));
        scrollView.addView(layout);
        setContentView(scrollView);

        append("LG G8X real-device validation benchmark\n");
        append("Warmup runs: " + WARMUP_RUNS + ", measured runs: " + MEASURED_RUNS + "\n");
        append("Press Run benchmark after generated .tflite assets are present.\n");
    }

    private void runBenchmarkAsync() {
        runButton.setEnabled(false);
        output.setText("");
        append("Starting benchmark...\n");
        new Thread(() -> {
            try {
                List<ModelSpec> specs = loadModelIndex();
                List<ResultRow> results = new ArrayList<>();
                for (ModelSpec spec : specs) {
                    ResultRow result = benchmarkModel(spec);
                    results.add(result);
                    appendOnUiThread(result.toHumanString() + "\n");
                }
                File outputFile = writeResults(results);
                appendOnUiThread("\nSaved CSV to:\n" + outputFile.getAbsolutePath() + "\n");
                Log.i(TAG, "Saved CSV to: " + outputFile.getAbsolutePath());
            } catch (Exception error) {
                Log.e(TAG, "Benchmark failed", error);
                appendOnUiThread("ERROR: " + error.getMessage() + "\n");
            } finally {
                runOnUiThread(() -> runButton.setEnabled(true));
            }
        }).start();
    }

    private List<ModelSpec> loadModelIndex() throws Exception {
        String json = readAssetText("model_index.json");
        JSONArray array = new JSONArray(json);
        List<ModelSpec> specs = new ArrayList<>();
        for (int i = 0; i < array.length(); i++) {
            JSONObject item = array.getJSONObject(i);
            specs.add(new ModelSpec(item));
        }
        return specs;
    }

    private ResultRow benchmarkModel(ModelSpec spec) throws Exception {
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(1);

        try (Interpreter interpreter = new Interpreter(loadModelFile(spec.modelAsset), options)) {
            int[] inputShape = interpreter.getInputTensor(0).shape();
            int[] outputShape = interpreter.getOutputTensor(0).shape();
            ByteBuffer input = allocateFloatBuffer(inputShape);
            ByteBuffer outputBuffer = allocateFloatBuffer(outputShape);

            for (int i = 0; i < WARMUP_RUNS; i++) {
                input.rewind();
                outputBuffer.rewind();
                interpreter.run(input, outputBuffer);
            }

            Runtime runtime = Runtime.getRuntime();
            Debug.MemoryInfo memoryBefore = readMemoryInfo();
            long javaHeapBeforeKb = usedJavaHeapKb(runtime);
            long nativeHeapBeforeKb = Debug.getNativeHeapAllocatedSize() / 1024;
            long cpuStartMs = Process.getElapsedCpuTime();
            List<Double> timingsMs = new ArrayList<>();
            for (int i = 0; i < MEASURED_RUNS; i++) {
                input.rewind();
                outputBuffer.rewind();
                long startNs = System.nanoTime();
                interpreter.run(input, outputBuffer);
                long elapsedNs = System.nanoTime() - startNs;
                timingsMs.add(elapsedNs / 1_000_000.0);
                SystemClock.sleep(5);
            }
            long cpuElapsedMs = Process.getElapsedCpuTime() - cpuStartMs;
            long nativeHeapAfterKb = Debug.getNativeHeapAllocatedSize() / 1024;
            long javaHeapAfterKb = usedJavaHeapKb(runtime);
            Debug.MemoryInfo memoryAfter = readMemoryInfo();
            double measuredWallTotalMs = sum(timingsMs);
            double cpuTimePerRunMs = cpuElapsedMs / (double) MEASURED_RUNS;
            double cpuWallRatio = measuredWallTotalMs > 0.0 ? cpuElapsedMs / measuredWallTotalMs : 0.0;

            return new ResultRow(
                    spec,
                    mean(timingsMs),
                    median(timingsMs),
                    stddev(timingsMs),
                    quantile(timingsMs, 0.25),
                    quantile(timingsMs, 0.75),
                    quantile(timingsMs, 0.90),
                    Collections.min(timingsMs),
                    Collections.max(timingsMs),
                    cpuElapsedMs,
                    cpuTimePerRunMs,
                    cpuWallRatio,
                    memoryBefore.getTotalPss(),
                    memoryAfter.getTotalPss(),
                    memoryAfter.getTotalPss() - memoryBefore.getTotalPss(),
                    javaHeapBeforeKb,
                    javaHeapAfterKb,
                    javaHeapAfterKb - javaHeapBeforeKb,
                    nativeHeapBeforeKb,
                    nativeHeapAfterKb,
                    nativeHeapAfterKb - nativeHeapBeforeKb
            );
        }
    }

    private MappedByteBuffer loadModelFile(String assetPath) throws Exception {
        File tempFile = File.createTempFile("model", ".tflite", getCacheDir());
        try (InputStream input = getAssets().open(assetPath);
             FileOutputStream output = new FileOutputStream(tempFile)) {
            byte[] buffer = new byte[8192];
            int read;
            while ((read = input.read(buffer)) != -1) {
                output.write(buffer, 0, read);
            }
        }

        try (FileInputStream input = new FileInputStream(tempFile);
             FileChannel channel = input.getChannel()) {
            return channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size());
        }
    }

    private ByteBuffer allocateFloatBuffer(int[] shape) {
        int elements = 1;
        for (int dim : shape) {
            elements *= dim;
        }
        ByteBuffer buffer = ByteBuffer.allocateDirect(elements * 4);
        buffer.order(ByteOrder.nativeOrder());
        for (int i = 0; i < elements; i++) {
            buffer.putFloat((i % 23) / 23.0f);
        }
        buffer.rewind();
        return buffer;
    }

    private Debug.MemoryInfo readMemoryInfo() {
        Debug.MemoryInfo memoryInfo = new Debug.MemoryInfo();
        Debug.getMemoryInfo(memoryInfo);
        return memoryInfo;
    }

    private long usedJavaHeapKb(Runtime runtime) {
        return (runtime.totalMemory() - runtime.freeMemory()) / 1024;
    }

    private File writeResults(List<ResultRow> results) throws Exception {
        String outputName = "real_device_validation_" + currentDeviceId() + ".csv";
        File outputFile = new File(getExternalFilesDir(null), outputName);
        StringBuilder csv = new StringBuilder();
        csv.append("block_id,benchmark_device,real_device,latency_group,predicted_latency,")
                .append("measured_mean_ms,measured_median_ms,measured_std_ms,")
                .append("measured_q25_ms,measured_q75_ms,measured_q90_ms,")
                .append("measured_min_ms,measured_max_ms,")
                .append("cpu_total_ms,cpu_per_run_ms,cpu_wall_ratio,")
                .append("pss_before_kb,pss_after_kb,pss_delta_kb,")
                .append("java_heap_before_kb,java_heap_after_kb,java_heap_delta_kb,")
                .append("native_heap_before_kb,native_heap_after_kb,native_heap_delta_kb,")
                .append("input_h,input_w,cin,cout,expansion,kernel,stride,group,block_name\n");
        for (ResultRow result : results) {
            csv.append(result.toCsv()).append("\n");
        }
        try (FileOutputStream output = new FileOutputStream(outputFile)) {
            output.write(csv.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8));
        }
        return outputFile;
    }

    private String readAssetText(String path) throws Exception {
        StringBuilder builder = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(getAssets().open(path), java.nio.charset.StandardCharsets.UTF_8)
        )) {
            String line;
            while ((line = reader.readLine()) != null) {
                builder.append(line).append("\n");
            }
        }
        return builder.toString();
    }

    private static double mean(List<Double> values) {
        return sum(values) / values.size();
    }

    private static double sum(List<Double> values) {
        double sum = 0.0;
        for (double value : values) {
            sum += value;
        }
        return sum;
    }

    private static double median(List<Double> values) {
        return quantile(values, 0.5);
    }

    private static double stddev(List<Double> values) {
        double mean = mean(values);
        double sumSquaredDiff = 0.0;
        for (double value : values) {
            double diff = value - mean;
            sumSquaredDiff += diff * diff;
        }
        return Math.sqrt(sumSquaredDiff / values.size());
    }

    private static double quantile(List<Double> values, double probability) {
        List<Double> sorted = new ArrayList<>(values);
        Collections.sort(sorted);
        if (sorted.size() == 1) {
            return sorted.get(0);
        }
        double position = probability * (sorted.size() - 1);
        int lower = (int) Math.floor(position);
        int upper = (int) Math.ceil(position);
        if (lower == upper) {
            return sorted.get(lower);
        }
        double weight = position - lower;
        return sorted.get(lower) * (1.0 - weight) + sorted.get(upper) * weight;
    }

    private static String sanitizeFilePart(String value) {
        String sanitized = value == null ? "android_device" : value.toLowerCase();
        sanitized = sanitized.replaceAll("[^a-z0-9]+", "_");
        sanitized = sanitized.replaceAll("^_+|_+$", "");
        return sanitized.isEmpty() ? "android_device" : sanitized;
    }

    private static String currentDeviceId() {
        String model = Build.MODEL == null ? "" : Build.MODEL.toLowerCase();
        if (model.contains("lm-g850") || model.contains("lg g8x")) {
            return "lg_g8x_thinq";
        }
        if (model.contains("redmi note 9 pro")) {
            return "redmi_note_9_pro";
        }
        return sanitizeFilePart(Build.MODEL);
    }

    private void append(String text) {
        output.append(text);
    }

    private void appendOnUiThread(String text) {
        runOnUiThread(() -> append(text));
    }

    private static class ModelSpec {
        final int blockId;
        final String modelAsset;
        final String benchmarkDevice;
        final String realDevice;
        final double predictedLatency;
        final String latencyGroup;
        final int inputH;
        final int inputW;
        final int cin;
        final int cout;
        final int expansion;
        final int kernel;
        final int stride;
        final int group;
        final String blockName;

        ModelSpec(JSONObject item) throws Exception {
            blockId = item.getInt("block_id");
            modelAsset = item.getString("model_asset");
            benchmarkDevice = item.getString("benchmark_device");
            realDevice = item.getString("real_device");
            predictedLatency = item.getDouble("predicted_latency");
            latencyGroup = item.getString("latency_group");
            inputH = item.getInt("input_h");
            inputW = item.getInt("input_w");
            cin = item.getInt("cin");
            cout = item.getInt("cout");
            expansion = item.getInt("expansion");
            kernel = item.getInt("kernel");
            stride = item.getInt("stride");
            group = item.getInt("group");
            blockName = item.getString("block_name");
        }
    }

    private static class ResultRow {
        final ModelSpec spec;
        final double meanMs;
        final double medianMs;
        final double stdMs;
        final double q25Ms;
        final double q75Ms;
        final double q90Ms;
        final double minMs;
        final double maxMs;
        final long cpuTotalMs;
        final double cpuPerRunMs;
        final double cpuWallRatio;
        final int pssBeforeKb;
        final int pssAfterKb;
        final int pssDeltaKb;
        final long javaHeapBeforeKb;
        final long javaHeapAfterKb;
        final long javaHeapDeltaKb;
        final long nativeHeapBeforeKb;
        final long nativeHeapAfterKb;
        final long nativeHeapDeltaKb;

        ResultRow(
                ModelSpec spec,
                double meanMs,
                double medianMs,
                double stdMs,
                double q25Ms,
                double q75Ms,
                double q90Ms,
                double minMs,
                double maxMs,
                long cpuTotalMs,
                double cpuPerRunMs,
                double cpuWallRatio,
                int pssBeforeKb,
                int pssAfterKb,
                int pssDeltaKb,
                long javaHeapBeforeKb,
                long javaHeapAfterKb,
                long javaHeapDeltaKb,
                long nativeHeapBeforeKb,
                long nativeHeapAfterKb,
                long nativeHeapDeltaKb
        ) {
            this.spec = spec;
            this.meanMs = meanMs;
            this.medianMs = medianMs;
            this.stdMs = stdMs;
            this.q25Ms = q25Ms;
            this.q75Ms = q75Ms;
            this.q90Ms = q90Ms;
            this.minMs = minMs;
            this.maxMs = maxMs;
            this.cpuTotalMs = cpuTotalMs;
            this.cpuPerRunMs = cpuPerRunMs;
            this.cpuWallRatio = cpuWallRatio;
            this.pssBeforeKb = pssBeforeKb;
            this.pssAfterKb = pssAfterKb;
            this.pssDeltaKb = pssDeltaKb;
            this.javaHeapBeforeKb = javaHeapBeforeKb;
            this.javaHeapAfterKb = javaHeapAfterKb;
            this.javaHeapDeltaKb = javaHeapDeltaKb;
            this.nativeHeapBeforeKb = nativeHeapBeforeKb;
            this.nativeHeapAfterKb = nativeHeapAfterKb;
            this.nativeHeapDeltaKb = nativeHeapDeltaKb;
        }

        String toHumanString() {
            return "block " + spec.blockId
                    + " [" + spec.latencyGroup + "]"
                    + " predicted=" + format(spec.predictedLatency)
                    + " measured median=" + format(medianMs) + " ms"
                    + " cpu/run=" + format(cpuPerRunMs) + " ms"
                    + " pss=" + pssAfterKb + " KB";
        }

        String toCsv() {
            return String.join(
                    ",",
                    Arrays.asList(
                            String.valueOf(spec.blockId),
                            spec.benchmarkDevice,
                            currentDeviceId(),
                            spec.latencyGroup,
                            format(spec.predictedLatency),
                            format(meanMs),
                            format(medianMs),
                            format(stdMs),
                            format(q25Ms),
                            format(q75Ms),
                            format(q90Ms),
                            format(minMs),
                            format(maxMs),
                            String.valueOf(cpuTotalMs),
                            format(cpuPerRunMs),
                            format(cpuWallRatio),
                            String.valueOf(pssBeforeKb),
                            String.valueOf(pssAfterKb),
                            String.valueOf(pssDeltaKb),
                            String.valueOf(javaHeapBeforeKb),
                            String.valueOf(javaHeapAfterKb),
                            String.valueOf(javaHeapDeltaKb),
                            String.valueOf(nativeHeapBeforeKb),
                            String.valueOf(nativeHeapAfterKb),
                            String.valueOf(nativeHeapDeltaKb),
                            String.valueOf(spec.inputH),
                            String.valueOf(spec.inputW),
                            String.valueOf(spec.cin),
                            String.valueOf(spec.cout),
                            String.valueOf(spec.expansion),
                            String.valueOf(spec.kernel),
                            String.valueOf(spec.stride),
                            String.valueOf(spec.group),
                            spec.blockName
                    )
            );
        }

        private static String format(double value) {
            return String.format(java.util.Locale.US, "%.6f", value);
        }
    }
}
