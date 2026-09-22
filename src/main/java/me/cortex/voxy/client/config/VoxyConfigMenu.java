package me.cortex.voxy.client.config;

import me.cortex.voxy.client.RenderStatistics;
import me.cortex.voxy.client.config.SodiumConfigBuilder.*;
import me.cortex.voxy.client.VoxyClient;
import me.cortex.voxy.client.VoxyClientInstance;
import me.cortex.voxy.client.core.IGetVoxyRenderSystem;
import me.cortex.voxy.common.util.cpu.CpuLayout;
import me.cortex.voxy.commonImpl.VoxyCommon;
import net.caffeinemc.mods.sodium.api.config.ConfigEntryPoint;
import net.caffeinemc.mods.sodium.api.config.option.OptionFlag;
import net.caffeinemc.mods.sodium.api.config.option.OptionImpact;
import net.caffeinemc.mods.sodium.api.config.option.Range;
import net.caffeinemc.mods.sodium.api.config.structure.ConfigBuilder;
import net.minecraft.client.Minecraft;
import net.minecraft.network.chat.Component;
import net.minecraft.resources.Identifier;

public class VoxyConfigMenu implements ConfigEntryPoint {
    @Override
    public void registerConfigLate(ConfigBuilder B) {
        var CFG = VoxyConfig.CONFIG;

        var cc = B.registerModOptions("voxy", "Voxy", VoxyCommon.MOD_VERSION)
                .setIcon(Identifier.parse("voxy:icon.png"));

        SodiumConfigBuilder.buildToSodium(B, cc, CFG::save, postOp->{
                    postOp.register("voxy:update_threads", ()->{
                        var instance = VoxyCommon.getInstance();
                        if (instance != null) {
                            instance.updateDedicatedThreads();
                        }
                    }, "voxy:enabled");
                },
                new Page(Component.translatable("voxy.config.general"),
                        new Group(
                                new BoolOption(
                                        "voxy:enabled",
                                        Component.translatable("voxy.config.general.enabled"),
                                        ()->CFG.enabled, v->CFG.enabled=v)
                                        .setPostChangeRunner(c->{
                                            if (c) {
                                                if (VoxyClientInstance.isInGame) {
                                                    VoxyCommon.createInstance();
                                                    var vrsh = (IGetVoxyRenderSystem) Minecraft.getInstance().levelRenderer;
                                                    if (vrsh != null && CFG.enableRendering) {
                                                        vrsh.createRenderer();
                                                    }
                                                }
                                            } else {
                                                var vrsh = (IGetVoxyRenderSystem) Minecraft.getInstance().levelRenderer;
                                                if (vrsh != null) {
                                                    vrsh.shutdownRenderer();
                                                }
                                                VoxyCommon.shutdownInstance();
                                            }
                                        }).setEnabler(null)
                        ), new Group(
                                new IntOption(
                                        "voxy:thread_count",
                                        Component.translatable("voxy.config.general.serviceThreads"),
                                        ()->CFG.serviceThreads, v->CFG.serviceThreads=v,
                                        new Range(1, CpuLayout.getCoreCount(), 1))
                                        .setPostChangeFlags("voxy:update_threads"),
                                new BoolOption(
                                        "voxy:use_sodium_threads",
                                        Component.translatable("voxy.config.general.useSodiumBuilder"),
                                        ()->!CFG.dontUseSodiumBuilderThreads, v->CFG.dontUseSodiumBuilderThreads=!v)
                                        .setPostChangeFlags("voxy:update_threads")
                        ), new Group(
                                new BoolOption(
                                        "voxy:ingest_enabled",
                                        Component.translatable("voxy.config.general.ingest"),
                                        ()->CFG.ingestEnabled, v->CFG.ingestEnabled=v)
                        )
                ).setEnabler("voxy:enabled"),
                new Page(Component.translatable("voxy.config.rendering"),
                        new Group(
                                new BoolOption(
                                        "voxy:rendering",
                                        Component.translatable("voxy.config.general.rendering"),
                                        ()->CFG.enableRendering, v->CFG.enableRendering=v)
                                        .setPostChangeRunner(c->{
                                            var vrsh = (IGetVoxyRenderSystem)Minecraft.getInstance().levelRenderer;
                                            if (vrsh != null) {
                                                if (c) {
                                                    vrsh.createRenderer();
                                                } else {
                                                    vrsh.shutdownRenderer();
                                                }
                                            }
                                        },"voxy:enabled", "voxy:renderer_reload")
                                        .setEnabler("voxy:enabled")
                        ), new Group(
                                new IntOption(
                                        "voxy:subdivsize",
                                        Component.translatable("voxy.config.general.subDivisionSize"),
                                        ()->subDivToSlider(CFG.subDivisionSize), v->CFG.subDivisionSize=sliderToSubDiv(v),
                                        new Range(0, SUBDIV_IN_MAX, 1))
                                        .setFormatter(v->Component.literal(Integer.toString(Math.round(sliderToSubDiv(v))))),
                                new IntOption(
                                        "voxy:render_distance",
                                        Component.translatable("voxy.config.general.renderDistance"),
                                        ()->CFG.sectionRenderDistance, v->CFG.sectionRenderDistance=v,
                                        new Range(2, 64, 1))
                                        .setFormatter(v->Component.literal(Integer.toString(v*32)))//Top level rd == 32 chunks
                                        .setPostChangeRunner(c->{
                                            var vrsh = (IGetVoxyRenderSystem)Minecraft.getInstance().levelRenderer;
                                            if (vrsh != null) {
                                                var vrs = vrsh.getVoxyRenderSystem();
                                                if (vrs != null) {
                                                    vrs.setRenderDistance(c);
                                                }
                                            }
                                        }, "voxy:rendering", "voxy:renderer_reload")
                        ), new Group(
                                new BoolOption(
                                        "voxy:eviromental_fog",
                                        Component.translatable("voxy.config.general.environmental_fog"),
                                        ()->CFG.useEnvironmentalFog, v->CFG.useEnvironmentalFog=v)
                                        .setPostChangeFlags(OptionFlag.REQUIRES_RENDERER_RELOAD.getId().toString())
                        ), new Group(
                                new BoolOption(
                                        "voxy:render_debug",
                                        Component.translatable("voxy.config.general.render_statistics"),
                                        ()-> RenderStatistics.enabled, v->RenderStatistics.enabled=v)
                                        .setPostChangeFlags(OptionFlag.REQUIRES_RENDERER_RELOAD.getId().toString()))
                ).setEnablerAND("voxy:enabled", "voxy:rendering"));

    }


    private static final int SUBDIV_IN_MAX = 100;
    private static final double SUBDIV_MIN = 28;
    /**
     * Raised 256 -> 1024 for HIDPI. The value is "maximum screen-space AABB area in pixels^2 before
     * subdividing", and the viewport it is measured against is the PHYSICAL backbuffer -- 1708x960 for
     * an 854x480 window on this Retina display. So a node at 64 may cover 4096 physical px^2, which is a
     * 32x32 LOGICAL square: four times finer on screen than the same setting on a 1x display. Matching
     * a 1x display's 64 needs 128 here, its 256 needs 512, and 1024 goes coarser than any 1x setting --
     * which is the range the top of the slider was missing.
     *
     * <p>Anything above this is still treated as drift by VoxyRenderSystem's startup guard; that guard's
     * threshold is kept in step with this constant, because it used to sit BELOW the slider's maximum
     * and would silently reset a deliberately-chosen high value back to 64 on the next launch.
     */
    private static final double SUBDIV_MAX = 1024;

    /**
     * Slider position -> value, QUADRATIC: {@code pos^2} spread across [SUBDIV_MIN, SUBDIV_MAX].
     *
     * <p>0 -> 28, 50 -> 277, 100 -> 1024. The previous mapping was log2, which gives every slider step
     * the same RATIO. A quadratic instead gives the top of the range finer absolute steps and the bottom
     * coarser ones -- roughly 2 units of value per slider step at 28, 10 at 277 and 20 at 1024, against
     * log2's 1, 10 and 37.
     *
     * <p>Nothing about the value's meaning changed; it is still "maximum screen-space AABB area in
     * pixels^2 before subdividing". Only how the slider travels across its range.
     */
    private static float sliderToSubDiv(int in) {
        double t = (double) in / SUBDIV_IN_MAX;
        return (float) (SUBDIV_MIN + (SUBDIV_MAX - SUBDIV_MIN) * t * t);
    }

    /**
     * Inverse of {@link #sliderToSubDiv}.
     *
     * <p>CLAMPED, and that is load-bearing rather than defensive. A config file can hold a value outside
     * [SUBDIV_MIN, SUBDIV_MAX] -- a hand edit, or a value written by an older build with a different
     * range -- and the log2 version this replaces did not clamp: a value below SUBDIV_MIN made the log
     * negative, which landed the slider at a negative position, and it is only the Range's own clamping
     * that stopped that becoming a visible misbehaviour. Under a square root a negative argument is
     * worse, not better: {@code (int) Math.sqrt(-x)} is NaN converted to int, i.e. 0, which would place
     * the slider at the bottom and misreport the value it is actually configured with.
     */
    private static int subDivToSlider(float value) {
        double t = ((double) value - SUBDIV_MIN) / (SUBDIV_MAX - SUBDIV_MIN);
        t = Math.max(0.0, Math.min(1.0, t));
        return (int) Math.round(Math.sqrt(t) * SUBDIV_IN_MAX);
    }
}
