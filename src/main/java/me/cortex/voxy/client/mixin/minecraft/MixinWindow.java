package me.cortex.voxy.client.mixin.minecraft;

import com.mojang.blaze3d.platform.DisplayData;
import com.mojang.blaze3d.platform.MonitorManager;
import com.mojang.blaze3d.platform.Window;
import com.mojang.blaze3d.platform.WindowEventHandler;
import me.cortex.voxy.client.GPUSelectorWindows2;
import me.cortex.voxy.common.util.ThreadUtils;
import org.lwjgl.glfw.GLFW;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

@Mixin(Window.class)
public class MixinWindow {
    /**
     * {@code -Dvoxy.noFocusOnShow=true}: stop the game window taking focus when it appears.
     *
     * <p>Automated capture runs are launched from a shell while someone is working in another
     * window, and a dev client that grabs focus on every launch makes that unusable. Minecraft
     * creates the window hidden ({@code GLFW_VISIBLE=false}) and shows it afterwards, so the focus
     * grab happens at show time and is governed by the {@code GLFW_FOCUS_ON_SHOW} hint — which has
     * to be set before {@code glfwCreateWindow}, hence injecting at the call rather than in the
     * constructor: the constructor runs before the backend sets its own hints.
     *
     * <p>The window is still created, shown and rendered normally; it just does not become the key
     * window. Screenshots are read back from the framebuffer, so they do not need focus either.
     */
    @Inject(method = "createGlfwWindow", at = @At(value = "INVOKE",
            target = "Lorg/lwjgl/glfw/GLFW;glfwCreateWindow(IILjava/lang/CharSequence;JJ)J"))
    private static void voxy$noFocusOnShow(final int width, final int height, final String title,
                                           final long monitor, final com.mojang.blaze3d.systems.GpuBackend backend,
                                           final CallbackInfoReturnable<Long> cir) {
        if (Boolean.getBoolean("voxy.noFocusOnShow")) {
            GLFW.glfwWindowHint(GLFW.GLFW_FOCUS_ON_SHOW, GLFW.GLFW_FALSE);
        }
    }

    @Inject(method = "<init>", at = @At(value = "INVOKE", target = "Lcom/mojang/blaze3d/platform/Window;setBootErrorCallback()V"))
    private void injectInitWindow(WindowEventHandler eventHandler, DisplayData settings, String fullscreenVideoMode, boolean fullscreen, String title, MonitorManager monitorManager, com.mojang.blaze3d.systems.GpuBackend backend, CallbackInfo ci) {
        //System.load("C:\\Program Files\\RenderDoc\\renderdoc.dll");
        var prop = System.getProperty("voxy.forceGpuSelectionIndex", "NO");
        if (!prop.equals("NO")) {
            GPUSelectorWindows2.doSelector(Integer.parseInt(prop));
        }

        //Force the current thread priority to be realtime
        Thread.currentThread().setPriority(Thread.MAX_PRIORITY);
        ThreadUtils.SetSelfThreadPriorityWin32(ThreadUtils.WIN32_THREAD_PRIORITY_TIME_CRITICAL);
    }
}
