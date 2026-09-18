package me.cortex.voxy.client.core.gpu;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Guards the threadgroup-size contract.
 *
 * <p>Metal dispatches the {@code threadsPerThreadgroup} the pipeline descriptor declares, whereas
 * GL uses the shader-declared size. When the two drifted (desc said 32, shaders said 128/256) only
 * ~25% of sections got draw commands, re-scrambled every frame — the 2026-05 LOD flicker. This
 * parser makes the shader authoritative, so its resolution rules are worth pinning down.
 */
class ComputeLocalSizeParserTest {

    @Test
    void parsesPlainDeclaration() {
        String glsl = "layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;\nvoid main() {}";
        ComputeLocalSizeParser.LocalSize size = ComputeLocalSizeParser.parse(glsl, Map.of(), "t");
        assertEquals(128, size.x());
        assertEquals(1, size.y());
        assertEquals(1, size.z());
    }

    @Test
    void defaultsMissingYAndZToOne() {
        String glsl = "layout(local_size_x=256) in;";
        ComputeLocalSizeParser.LocalSize size = ComputeLocalSizeParser.parse(glsl, Map.of(), "t");
        assertEquals(256, size.x());
        assertEquals(1, size.y(), "undeclared local_size_y must default to 1, not 0");
        assertEquals(1, size.z(), "undeclared local_size_z must default to 1, not 0");
    }

    @Test
    void resolvesSizeFromSuppliedDefines() {
        // The real shape: the layout names a macro, the shader defines it.
        String glsl = "#define LOCAL_SIZE 128\nlayout(local_size_x = LOCAL_SIZE) in;";
        ComputeLocalSizeParser.LocalSize size = ComputeLocalSizeParser.parse(glsl, Map.of(), "t");
        assertEquals(128, size.x());
    }

    @Test
    void resolvesSizeFromDefinesMapOverridingSource() {
        String glsl = "#define LOCAL_SIZE 32\nlayout(local_size_x = LOCAL_SIZE) in;";
        ComputeLocalSizeParser.LocalSize size =
                ComputeLocalSizeParser.parse(glsl, Map.of("LOCAL_SIZE", "256"), "t");
        assertEquals(256, size.x(), "desc defines must win over the in-source #define");
    }

    @Test
    void resolvesConstantIntegerExpressionThroughMacro() {
        // traversal_dev.comp's real shape: the layout names a macro, and the macro's body is the
        // expression. The expression must NOT be written inline in the layout — the declaration
        // regex is [^)]*, which deliberately assumes size expressions live behind macro tokens
        // (see the parser's note). Resolving them is the evaluator's job.
        String glsl = "#define LOCAL_SIZE_BITS 7\n"
                + "#define LOCAL_SIZE (1 << LOCAL_SIZE_BITS)\n"
                + "layout(local_size_x = LOCAL_SIZE) in;";
        ComputeLocalSizeParser.LocalSize size = ComputeLocalSizeParser.parse(glsl, Map.of(), "t");
        assertEquals(128, size.x());
    }

    @Test
    void throwsOnMissingDeclaration() {
        // Must never dispatch with a guessed size.
        assertThrows(IllegalStateException.class,
                () -> ComputeLocalSizeParser.parse("void main() {}", Map.of(), "t"));
    }

    @Test
    void throwsOnNullSource() {
        assertThrows(IllegalArgumentException.class,
                () -> ComputeLocalSizeParser.parse(null, Map.of(), "t"));
    }

    @Test
    void ignoresNonComputeLayoutQualifiers() {
        // A vertex-stage layout must not be mistaken for the compute declaration.
        String glsl = "layout(location = 0) in vec3 pos;\nlayout(local_size_x = 64) in;";
        ComputeLocalSizeParser.LocalSize size = ComputeLocalSizeParser.parse(glsl, Map.of(), "t");
        assertEquals(64, size.x());
    }
}
