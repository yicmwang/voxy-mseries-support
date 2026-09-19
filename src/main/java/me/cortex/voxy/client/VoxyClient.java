package me.cortex.voxy.client;

import me.cortex.voxy.client.core.IGetVoxyRenderSystem;
import me.cortex.voxy.client.core.VoxyRenderSystem;
import me.cortex.voxy.client.core.gl.Capabilities;
import me.cortex.voxy.client.core.gpu.RenderBackend;
import me.cortex.voxy.client.core.gpu.RenderBackendFactory;
import me.cortex.voxy.client.core.model.bakery.BudgetBufferRenderer;
import me.cortex.voxy.client.core.rendering.util.SharedIndexBuffer;
import me.cortex.voxy.common.Logger;
import me.cortex.voxy.commonImpl.VoxyCommon;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.ChatFormatting;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.components.debug.DebugScreenDisplayer;
import net.minecraft.client.gui.components.debug.DebugScreenEntries;
import net.minecraft.client.gui.components.debug.DebugScreenEntry;
import net.minecraft.resources.Identifier;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.chunk.LevelChunk;
import org.jspecify.annotations.Nullable;

import java.util.HashSet;
import java.util.function.Consumer;
import java.util.function.Function;

public class VoxyClient implements ClientModInitializer {
    private static final HashSet<String> FREX = new HashSet<>();

    public static void initVoxyClient() {
        Capabilities.init();//Ensure clinit is called

        if (Capabilities.INSTANCE.hasBrokenDepthSampler) {
            Logger.error("AMD broken depth sampler detected, voxy does not work correctly and has been disabled, this will hopefully be fixed in the future");
        }

        // Query the active render backend instead of OpenGL capabilities directly:
        // on macOS Apple Silicon the OpenGL driver tops out at 4.1 (no compute,
        // no indirect draws), which would fail this check even though the Metal
        // backend supports both. MetalRenderBackend reports compute and indirect
        // as available; the OpenGL backend delegates to the legacy Capabilities.
        RenderBackend backend = RenderBackendFactory.get();
        Logger.info("Render backend: " + backend.getType()
                + " (compute=" + backend.hasCompute()
                + ", indirectParameters=" + backend.hasIndirectParameters() + ")");

        boolean systemSupported = backend.hasCompute() && backend.hasIndirectParameters() && !Capabilities.INSTANCE.hasBrokenDepthSampler;

        // M9 transitional: even though MetalRenderBackend reports compute=true and
        // indirectParameters=true, Voxy's render path (MDICSectionRenderer,
        // HiZBuffer2, HierarchicalOcclusionTraverser, ChunkBoundRenderer,
        // bakery, AbstractRenderPipeline) is still raw OpenGL DSA and aborts
        // the JVM the first time the GL driver hits an unsupported call on
        // Apple's frozen GL 4.1. Until that work lands (M9-M11 migration to
        // the encoder API + IOSurface bridge), force-disable Voxy on
        // non-OpenGL backends so the mixins find VoxyRenderSystem == null
        // and Sodium's chunk render runs unmodified — the user sees normal
        // close-distance MC + Sodium rendering instead of a blue screen.
        // Feature flag to opt into the Metal render path. The flag exists
        // separately from regular env vars so opting in is intentional —
        // until the full MDIC encoder migration lands, this path produces
        // a clear color via the IOSurfaceBridge, not actual LOD chunks.
        boolean forceMetal = "1".equals(System.getenv("VOXY_FORCE_METAL"))
                || "true".equals(System.getenv("VOXY_FORCE_METAL"))
                || "true".equals(System.getProperty("voxy.forceMetal", "false"));

        if (systemSupported && backend.getType() != me.cortex.voxy.client.core.gpu.BackendType.OPENGL) {
            if (forceMetal) {
                Logger.info("[VOXY_FORCE_METAL] Voxy enabled on " + backend.getType()
                        + " backend. Under Metallum the LOD pass renders straight into the "
                        + "frame's colour and depth attachments; there is no bridge and no "
                        + "composite.");
            } else {
                Logger.warn("[M9 TRANSITIONAL] Voxy disabled on " + backend.getType()
                        + " backend. Set VOXY_FORCE_METAL=1 to enable the Metal render path.");
                systemSupported = false;
            }
        }

        if (systemSupported) {

            if (backend.getType() == me.cortex.voxy.client.core.gpu.BackendType.OPENGL) {
                // Both of these are raw-GL startup: id() creates the GL index buffer and
                // BudgetBufferRenderer compiles a GL program. Under whole-frame Metal there is no
                // GL context at all, and glCreateShader does not fail gracefully -- LWJGL aborts
                // the JVM ("FATAL ERROR in native method: No context is current"). The Metal paths
                // allocate their own resources lazily, and MetalBudgetBufferRenderer is dormant.
                SharedIndexBuffer.INSTANCE.id();
                BudgetBufferRenderer.init();
            }

            VoxyCommon.setInstanceFactory(VoxyClientInstance::new);

            if (!Capabilities.INSTANCE.subgroup) {
                Logger.warn("GPU does not support subgroup operations, expect some performance degradation");
            }

        } else {
            Logger.error("Voxy is unsupported on your system.");
        }
    }

    @Override
    public void onInitializeClient() {
        // Iris-pack + Metal coexistence now happens at the SOLID-pass head:
        // IrisGbufferInjector draws the LOD bridge into the pack's terrain
        // gbuffer (MixinDefaultChunkRenderer). The former HUD-time late
        // composite that lived here is gone — it ran after Iris finalized but
        // painted over the pack's post chain and skipped entirely with the
        // HUD hidden (F1).
        DebugScreenEntries.register(Identifier.fromNamespaceAndPath("voxy", "version"), new DebugScreenEntry() {
            @Override
            public void display(DebugScreenDisplayer lines, @Nullable Level level, @Nullable LevelChunk levelChunk, @Nullable LevelChunk levelChunk2) {
                if (!VoxyCommon.isAvailable()) {
                    lines.addLine(ChatFormatting.RED + "voxy-"+VoxyCommon.MOD_VERSION);//Voxy installed, not avalible
                    return;
                }
                var instance = VoxyCommon.getInstance();
                if (instance == null) {
                    lines.addLine(ChatFormatting.YELLOW + "voxy-" + VoxyCommon.MOD_VERSION);//Voxy avalible, no instance active
                    return;
                }
                VoxyRenderSystem vrs = null;
                var wr = Minecraft.getInstance().levelRenderer;
                if (wr != null) vrs = ((IGetVoxyRenderSystem) wr).getVoxyRenderSystem();

                //Voxy instance active
                lines.addLine((vrs==null?ChatFormatting.DARK_GREEN:ChatFormatting.GREEN)+"voxy-"+VoxyCommon.MOD_VERSION);
            }
        });

        DebugScreenEntries.register(Identifier.fromNamespaceAndPath("voxy","debug"), new VoxyDebugScreenEntry());
        ClientCommandRegistrationCallback.EVENT.register((dispatcher, registryAccess) -> {
            if (VoxyCommon.isAvailable()) {
                dispatcher.register(VoxyCommands.register());
            }
        });

        // VOXY_AUTO_SCREENSHOT=<seconds>: periodically save an in-game
        // screenshot via MC's own Screenshot API (lands in run/screenshots).
        // Debug-loop aid — headless verification can capture frames without
        // desktop screencapture (which fails when other windows are
        // frontmost on the test machine).
        String autoShot = System.getenv("VOXY_AUTO_SCREENSHOT");
        if (autoShot != null && !autoShot.isBlank()) {
            int parsedInterval;
            try {
                parsedInterval = Math.max(2, Integer.parseInt(autoShot.trim()));
            } catch (NumberFormatException e) {
                parsedInterval = 10;
            }
            final long intervalNanos = parsedInterval * 1_000_000_000L;
            final long[] last = {System.nanoTime()};
            net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents.END_CLIENT_TICK.register(client -> {
                if (client.level == null || client.gameRenderer.mainRenderTarget() == null) return;
                long now = System.nanoTime();
                if (now - last[0] < intervalNanos) return;
                last[0] = now;
                net.minecraft.client.Screenshot.grab(client.gameDirectory,
                        client.gameRenderer.mainRenderTarget(), component -> {});
            });
            Logger.info("VOXY_AUTO_SCREENSHOT active: every " + parsedInterval + "s");
        }

        // VOXY_DEV_TIME=<time>: pin the world clock a few seconds after joining, so screenshots are
        // comparable. Debug-loop aid.
        //
        // Without this, brightness comparisons are meaningless: the day/night cycle runs at 20 real
        // minutes per in-game day, so two runs an hour apart land at arbitrary times and "the LOD
        // renders black" is indistinguishable from "it is night". A black frame with a bright hotbar
        // is exactly the night signature, and that ambiguity cost a round of diagnosis.
        //
        // Accepts any /time set argument ("noon", "midnight", "day", or a tick count), and also
        // clears the weather, which darkens the sky the same way.
        String devTime = System.getenv("VOXY_DEV_TIME");
        if (devTime != null && !devTime.isBlank()) {
            final String timeArg = devTime.trim();
            final int[] ticksUntilApply = {40};   // let the world finish loading first
            final boolean[] applied = {false};
            net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents.END_CLIENT_TICK.register(client -> {
                if (applied[0] || client.level == null) return;
                if (ticksUntilApply[0]-- > 0) return;
                var server = client.getSingleplayerServer();
                if (server == null) return;
                applied[0] = true;
                try {
                    var source = server.createCommandSourceStack();
                    var commands = server.getCommands();
                    for (String cmd : new String[] {
                            "gamerule doDaylightCycle false",
                            "gamerule doWeatherCycle false",
                            "weather clear",
                            "time set " + timeArg}) {
                        commands.performPrefixedCommand(source, cmd);
                    }
                    Logger.info("VOXY_DEV_TIME active: pinned the world clock to '" + timeArg
                            + "' with the daylight cycle off");
                } catch (Throwable t) {
                    Logger.error("VOXY_DEV_TIME failed to apply '" + timeArg + "'", t);
                }
            });
        }

        // VOXY_FPS_LOG=<seconds>: log Minecraft's own frame rate.
        //
        // Voxy's frame-rate diagnostic (Metal-RING) only exists when Voxy's render system is
        // running, so it cannot measure the vanilla-only baseline of an A/B -- which is exactly the
        // measurement that decides whether a slowdown belongs to Voxy or to what it renders into.
        // This reads MC's own counter, so it reports either way.
        String fpsLog = System.getenv("VOXY_FPS_LOG");
        if (fpsLog != null && !fpsLog.isBlank()) {
            final long periodNanos;
            try {
                periodNanos = Math.max(1, Long.parseLong(fpsLog.trim())) * 1_000_000_000L;
            } catch (NumberFormatException e) {
                throw new IllegalStateException("VOXY_FPS_LOG must be an integer number of seconds", e);
            }
            final long[] last = {System.nanoTime()};
            net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents.END_CLIENT_TICK.register(client -> {
                long now = System.nanoTime();
                if (now - last[0] < periodNanos) return;
                last[0] = now;
                Logger.info("[FPS] " + client.getFps()
                        + (client.level == null ? "  (no world)" : "  chunks=" + client.level.getChunkSource().getLoadedChunksCount()));
            });
        }

        // VOXY_TEST_CAMERA="x,y,z,yaw,pitch": pin the player's position and facing, in flight,
        // re-applied every tick so server corrections cannot drift the view.
        //
        // Position is what makes a capture comparable at all. Two frames of the same build — or of
        // two builds — can only be diffed if the camera is identical, and the alternative (issuing
        // /tp) is unreliable: Minecraft rewrites level.dat's allowCommands on save, so a world
        // patched to allow cheats silently reverts and every command comes back
        // "Unknown or incomplete command". Driving the entity directly sidesteps world settings
        // entirely, and works on a server too.
        String testCamera = System.getenv("VOXY_TEST_CAMERA");
        if (testCamera != null && !testCamera.isBlank()) {
            String[] parts = testCamera.split(",");
            if (parts.length != 5) {
                throw new IllegalStateException(
                        "VOXY_TEST_CAMERA needs \"x,y,z,yaw,pitch\"; got: " + testCamera);
            }
            final double cx = Double.parseDouble(parts[0].trim());
            final double cy = Double.parseDouble(parts[1].trim());
            final double cz = Double.parseDouble(parts[2].trim());
            final float cyaw = Float.parseFloat(parts[3].trim());
            final float cpitch = Float.parseFloat(parts[4].trim());
            net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents.END_CLIENT_TICK.register(client -> {
                if (client.player == null) return;
                var p = client.player;
                p.setPos(cx, cy, cz);
                p.setYRot(cyaw);
                p.setXRot(cpitch);
                p.setDeltaMovement(0, 0, 0);
                p.resetFallDistance();
                var ab = p.getAbilities();
                ab.mayfly = true;
                ab.flying = true;
            });
            Logger.info("VOXY_TEST_CAMERA active: " + testCamera + " (flying, re-pinned each tick)");
        }

        // VOXY_JOIN_COMMAND="/time set noon; /weather clear; /tp 0 120 0 0 30":
        // run slash-commands once a few seconds after joining a world, then stop.
        //
        // This is what makes a capture *comparable*. Comparing two builds — or two
        // runs of one build — only means something if both rendered the same scene,
        // and time of day, weather, position and facing are otherwise whatever the
        // save happened to be left holding. Pairs with VOXY_AUTO_SCREENSHOT; needs
        // allowCommands on the world (a test fixture, not a gameplay world).
        String joinCommand = System.getenv("VOXY_JOIN_COMMAND");
        if (joinCommand != null && !joinCommand.isBlank()) {
            final int delayTicks;
            try {
                delayTicks = Math.max(20, Integer.parseInt(
                        System.getenv().getOrDefault("VOXY_JOIN_COMMAND_DELAY", "60").trim()));
            } catch (NumberFormatException e) {
                throw new IllegalStateException("VOXY_JOIN_COMMAND_DELAY must be an integer", e);
            }
            final String[] commands = joinCommand.split(";");
            final int[] ticks = {0};
            final boolean[] done = {false};
            net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents.END_CLIENT_TICK.register(client -> {
                if (done[0] || client.level == null || client.player == null
                        || client.getConnection() == null) {
                    return;
                }
                if (++ticks[0] < delayTicks) return;
                done[0] = true;
                for (String raw : commands) {
                    String cmd = raw.trim();
                    if (cmd.isEmpty()) continue;
                    // sendCommand takes the body without the leading slash.
                    client.getConnection().sendCommand(cmd.startsWith("/") ? cmd.substring(1) : cmd);
                }
                Logger.info("VOXY_JOIN_COMMAND ran after " + delayTicks + " ticks: " + joinCommand);
            });
        }

        FabricLoader.getInstance()
                .getEntrypoints("frex_flawless_frames", Consumer.class)
                .forEach(api -> ((Consumer<Function<String,Consumer<Boolean>>>)api).accept(name->active->{if (active) {
                    FREX.add(name);
                } else {
                    FREX.remove(name);
                }}));
    }

    public static boolean isFrexActive() {
        return !FREX.isEmpty();
    }

    public static int getOcclusionDebugState() {
        return 0;
    }

    public static boolean disableSodiumChunkRender() {
        return false;// getOcclusionDebugState() != 0;
    }
}
