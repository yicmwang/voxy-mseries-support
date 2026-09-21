package me.cortex.voxy.client.core.gpu.shader;


import net.minecraft.resources.Identifier;
import org.apache.commons.io.IOUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

public class ShaderLoader {
    public static String parse(String id) {
        var src =  "#version 460 core\n";
        src += String.join("\n", ShaderLoadingParser.parseRoot(Identifier.parse(id)));
        // Strip printf calls unless printf debugging is on. Voxy's debug helpers in node.glsl /
        // queue.glsl declare them unconditionally, and glslang rejects `printf(string-literal, ...)`
        // unless GL_EXT_debug_printf is requested -- which the runtime SPIRV/MSL path does not do.
        // Without this the compute pipelines fail to transpile and the renderer never starts.
        // (Upstream does this via PrintfInjector; this loader keeps the simpler substitution.)
        if (!me.cortex.voxy.client.core.rendering.util.PrintfDebugUtil.ENABLE_PRINTF_DEBUGGING) {
            src = src.replace("printf", "//printf");
        }
        return src;
    }


    //Use our own loader

    private static final class ShaderLoadingParser {
        private static final Pattern IMPORT_PATTERN = Pattern.compile("#import <(?<namespace>.*):(?<path>.*)>");
        public static List<String> parseRoot(Identifier id) {
            List<String> out = new ArrayList<>();
            for (var line : toLines(loadShaderAsset(id))) {
                if (line.startsWith("#version")) {
                    continue;
                } else if (line.startsWith("#import")) {
                    var match = IMPORT_PATTERN.matcher(line);
                    if (!match.matches()) throw new IllegalArgumentException("Unknown import: " + line);
                    var iid = Identifier.fromNamespaceAndPath(match.group("namespace"), match.group("path"));
                    out.addAll(parseRoot(iid));
                } else {
                    out.add(line);
                }
            }
            return out;
        }

        private static List<String> toLines(String src) {
            try {
                return new BufferedReader(new StringReader(src)).readAllLines();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
        private static String loadShaderAsset(Identifier id) {
            String path = String.format("/assets/%s/shaders/%s", id.getNamespace(), id.getPath());
            try (InputStream in = ShaderLoadingParser.class.getResourceAsStream(path)) {
                if (in == null) {
                    throw new RuntimeException("Shader not found: " + path);
                } else {
                    return IOUtils.toString(in, StandardCharsets.UTF_8);
                }
            } catch (IOException e) {
                throw new RuntimeException("Failed to read shader source for " + path, e);
            }
        }
    }
}
