package me.cortex.voxy.client.core.gpu.shader;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * RECORDS A CLOSED ROUTE, AND GUARDS IT.
 *
 * <p>Bug A's last 0.147 % is the pyramid's CONTENT, not its timing: the LOD pass writes
 * {@code gl_FragCoord.z} only for fragments that survive the depth test against vanilla terrain, so the
 * pyramid records what was <em>visible</em> rather than what was <em>drawn</em>. A node hidden behind
 * vanilla has no entry, so next frame its own box is tested against a footprint holding the near ground
 * and it gets culled. Upstream is immune because its pyramid holds MC's whole frame. Fixing it needs
 * the frame's depth in the pyramid.
 *
 * <p>Metal will not let a shader sample a Depth32Float through an ordinary sampler — it wants
 * {@code depth2d} — and the aliasing route is dead (Metal asserts Depth32Float is not view-compatible
 * with R32Float, and the assertion aborts the process). So the question was whether {@code depth2d} is
 * reachable from GLSL, and the candidate was {@code texelFetch} on a {@code sampler2DShadow}.
 *
 * <p><b>It is not.</b> glslang rejects the overload before SPIRV-Cross is reached:
 * <pre>voxy_shader:8: error: 'texelFetch' : no matching overloaded function found</pre>
 * and the only shadow-sampler reads that DO compile ({@code texture}, {@code textureLod},
 * {@code textureGather}) return a comparison result, 0 or 1, not the depth — useless for a min-reduce.
 *
 * <p>So the frame's depth is unreadable from GLSL by every route, and reading it requires a
 * hand-written MSL pipeline declaring {@code depth2d<float>} and calling {@code read(uint2)}.
 * {@code GraphicsPipelineDesc} already accepts hand-supplied MSL, so that is expressible; it is simply
 * real work rather than a binding.
 *
 * <p>This test asserts the REJECTION rather than deleting the probe, so that the finding is executable
 * rather than a comment: if a future toolchain starts accepting the overload, this test fails and says
 * the cheap route is open again.
 */
public class ShadowSamplerMslProbeTest {

    @Test
    public void texelFetchOnAShadowSamplerIsRejectedByGlslang() {
        String glsl = """
                #version 430

                layout(binding = 0) uniform sampler2DShadow frameDepth;
                layout(location = 0) out vec4 colour;

                void main() {
                    ivec2 p = ivec2(gl_FragCoord.xy);
                    float d = texelFetch(frameDepth, p, 0);
                    colour = vec4(d, d, d, 1.0);
                }
                """;

        RuntimeException e = assertThrows(RuntimeException.class, () ->
                RuntimeShaderCompiler.compile(glsl, RuntimeShaderCompiler.Stage.FRAGMENT, Map.of(),
                        RuntimeShaderCompiler.Target.METAL_MSL));

        String msg = String.valueOf(e.getMessage());
        assertTrue(msg.contains("texelFetch"),
                "expected glslang to reject texelFetch on sampler2DShadow; got: " + msg);
        assertTrue(msg.contains("no matching overloaded function"),
                "the rejection should be the overload, not something incidental; got: " + msg);
    }
}
