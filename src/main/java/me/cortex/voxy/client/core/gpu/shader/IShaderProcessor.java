package me.cortex.voxy.client.core.gpu.shader;

public interface IShaderProcessor {
    String process(ShaderType type, String source);
}
