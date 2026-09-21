package me.cortex.voxy.client.core.rendering.util;

import me.cortex.voxy.client.core.gpu.shader.IShaderProcessor;
import me.cortex.voxy.client.core.gpu.shader.PrintfInjector;
import me.cortex.voxy.client.core.gpu.shader.ShaderType;
import me.cortex.voxy.common.Logger;

import java.util.ArrayList;
import java.util.List;

public final class PrintfDebugUtil {
    public static final boolean ENABLE_PRINTF_DEBUGGING = System.getProperty("voxy.enableShaderDebugPrintf", "false").equals("true");

    private static final List<String> printfQueue2 = new ArrayList<>();
    private static final List<String> printfQueue = new ArrayList<>();
    public static final IShaderProcessor PRINTF_processor;
    private static final PrintfInjector PRINTF_object;


    static {
        if (ENABLE_PRINTF_DEBUGGING) {
            PRINTF_object = new PrintfInjector(50000, 20, line -> {
                if (line.startsWith("LOG")) {
                    Logger.info(line);
                }
                printfQueue.add(line);
            }, printfQueue::clear);
            PRINTF_processor = PRINTF_object;
        } else {
            PRINTF_object = null;
            //Todo add a dummy processor that just removes all the printf calls
            PRINTF_processor = new IShaderProcessor() {
                @Override
                public String process(ShaderType type, String src) {
                    //TODO: replace with https://stackoverflow.com/questions/47162098/is-it-possible-to-match-nested-brackets-with-a-regex-without-using-recursion-or/47162099#47162099
                    // to match on printf with balanced bracing
                    return src.replace("printf", "//printf");
                }
            };
        }
    }

    public static void tick() {
        if (ENABLE_PRINTF_DEBUGGING) {
            printfQueue2.clear();
            printfQueue2.addAll(printfQueue);
            printfQueue.clear();
            PRINTF_object.download();
        }
    }

    public static void addToOut(List<String> out) {
        if (ENABLE_PRINTF_DEBUGGING) {
            out.add("Printf Queue: ");
            out.addAll(printfQueue2);
        }
    }

    // `bind()` stood here. It did a raw `glBindBufferBase`, which is gone with the GL backend — and
    // that is the whole of the printf-output binding on this backend: there is no cross-backend way to
    // bind an SSBO outside an encoder. `tick()`/`addToOut()` and the shader source injection are
    // unaffected, so the feature still collects and prints if enabled; reviving the BIND means a
    // `ComputeEncoder.setBuffer` inside the traverser's compute pass, which is where the printf buffer
    // is read. Off by default (`-Dvoxy.enableShaderDebugPrintf=true`).
}
