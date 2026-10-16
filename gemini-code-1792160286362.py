bl_info = {
    "name": "OpenCV Inpaint Brush Multi-Node",
    "author": "AI Collaborator",
    "version": (1, 3, 0),
    "blender": (3, 6, 0),
    "location": "View3D / Image Editor > Sidebar (N) > Tool > OpenCV Inpaint",
    "description": "Maalaa poistettava alue; suorittaa OpenCV Inpaintin vasta kun hiiren painike vapautetaan",
    "category": "Paint",
}

import bpy
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


class InpaintState:
    active = False
    primary_image_name = ""
    target_image_names = []
    snapshots_float = {}
    undo_stack = {}
    prev_color = (1.0, 1.0, 1.0)
    prev_blend = 'MIX'
    prev_strength = 1.0


def get_target_paint_images(context):
    primary_img = None
    target_imgs = []

    if context.area and context.area.type == 'IMAGE_EDITOR':
        if context.space_data.image:
            primary_img = context.space_data.image

    ts = context.tool_settings.image_paint
    if not primary_img and ts.mode == 'IMAGE' and ts.canvas:
        primary_img = ts.canvas

    obj = context.active_object
    if obj and obj.active_material and obj.active_material.use_nodes:
        nodes = obj.active_material.node_tree.nodes

        for n in nodes:
            if n.type == 'TEX_IMAGE' and n.select and n.image:
                if n.image not in target_imgs:
                    target_imgs.append(n.image)

        if not primary_img:
            if nodes.active and nodes.active.type == 'TEX_IMAGE' and nodes.active.image:
                primary_img = nodes.active.image
            elif target_imgs:
                primary_img = target_imgs[0]
            else:
                for n in nodes:
                    if n.type == 'TEX_IMAGE' and n.image:
                        primary_img = n.image
                        break

    if primary_img:
        if primary_img in target_imgs:
            target_imgs.remove(primary_img)
        target_imgs.insert(0, primary_img)

    return primary_img, target_imgs


def save_snapshots(images):
    InpaintState.snapshots_float.clear()
    InpaintState.undo_stack.clear()
    InpaintState.target_image_names.clear()

    for img in images:
        w, h = img.size
        if w == 0 or h == 0:
            continue
        pixels = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(pixels)
        arr = pixels.reshape((h, w, 4)).copy()
        InpaintState.snapshots_float[img.name] = arr
        InpaintState.target_image_names.append(img.name)

        img.pixels.foreach_set(pixels)
        img.update()


def inpaint_single_image(img, mask, scene):
    orig_float = InpaintState.snapshots_float.get(img.name)
    if orig_float is None:
        return

    w, h = img.size
    if (h, w, 4) != orig_float.shape:
        return

    # Tallennetaan tila ennen muokkausta Undo-toimintoa varten
    InpaintState.undo_stack[img.name] = orig_float.copy()

    if mask.shape[0] != h or mask.shape[1] != w:
        local_mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        local_mask = mask

    orig_rgb_8u = (np.clip(orig_float[:, :, :3], 0.0, 1.0) * 255.0).astype(np.uint8)

    algo = cv2.INPAINT_TELEA if scene.cv_inpaint_algo == 'TELEA' else cv2.INPAINT_NS
    radius = scene.cv_inpaint_radius

    inpainted_8u = cv2.inpaint(orig_rgb_8u, local_mask, radius, algo)

    result_float = orig_float.copy()
    mask_bool = local_mask > 0
    result_float[mask_bool, :3] = inpainted_8u[mask_bool].astype(np.float32) / 255.0

    img.pixels.foreach_set(result_float.ravel())
    img.update()
    InpaintState.snapshots_float[img.name] = result_float.copy()


def run_multi_opencv_inpaint(primary_img, scene):
    """Tarkistaa onko kuvaa maalattu ja suorittaa Inpaintin kaikille kohteille."""
    if not HAS_CV2:
        return False

    orig_primary = InpaintState.snapshots_float.get(primary_img.name)
    if orig_primary is None:
        return False

    w, h = primary_img.size
    if (h, w, 4) != orig_primary.shape:
        return False

    curr_pixels = np.empty(w * h * 4, dtype=np.float32)
    primary_img.pixels.foreach_get(curr_pixels)
    curr_float = curr_pixels.reshape((h, w, 4))

    # Verrataan nykyistä kuvaa pohjakuvaan
    diff = np.max(np.abs(curr_float[:, :, :3] - orig_primary[:, :, :3]), axis=2)
    mask = (diff > 0.02).astype(np.uint8) * 255

    # Jos mitään ei ole maalattu, poistutaan nopeasti
    if np.count_nonzero(mask) == 0:
        return False

    dilate_px = scene.cv_inpaint_dilate
    if dilate_px > 0:
        kernel = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    for img_name in InpaintState.target_image_names:
        target_img = bpy.data.images.get(img_name)
        if target_img:
            inpaint_single_image(target_img, mask, scene)

    return True


def restore_brush_settings(context):
    brush = context.tool_settings.image_paint.brush
    if brush:
        brush.color = InpaintState.prev_color
        brush.blend = InpaintState.prev_blend
        brush.strength = InpaintState.prev_strength


class PAINT_OT_cv_inpaint_watcher(bpy.types.Operator):
    """Modaalinen tarkkailija, joka odottaa maalausvedon päättymistä (hiiren vapautusta)"""
    bl_idname = "paint.cv_inpaint_watcher"
    bl_label = "OpenCV Inpaint Stroke Watcher"

    def modal(self, context, event):
        if not InpaintState.active:
            return {'FINISHED'}

        # Kun maalaat, Blenderin oma maalausoperaattori kaappaa hiiren liikkeen.
        # Heti kun vapautat hiiren painikkeen, maalausoperaattori päättyy ja tämä ehto laukeaa!
        if (event.type == 'LEFTMOUSE' and event.value == 'RELEASE') or event.type == 'MOUSEMOVE':
            primary_img = bpy.data.images.get(InpaintState.primary_image_name)
            if not primary_img:
                InpaintState.active = False
                restore_brush_settings(context)
                return {'FINISHED'}

            # Tarkistetaan ja ajetaan Inpaint vain, jos kuvaan on tullut valmis maalausveto
            if primary_img.is_dirty:
                if run_multi_opencv_inpaint(primary_img, context.scene):
                    for window in context.window_manager.windows:
                        for area in window.screen.areas:
                            area.tag_redraw()

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}


class PAINT_OT_toggle_cv_inpaint(bpy.types.Operator):
    bl_idname = "paint.toggle_cv_inpaint"
    bl_label = "Toggle OpenCV Inpaint Mode"

    def execute(self, context):
        if not HAS_CV2:
            self.report({'ERROR'}, "OpenCV (cv2) puuttuu! Asenna 'opencv-python' Blenderin Pythoniin.")
            return {'CANCELLED'}

        brush = context.tool_settings.image_paint.brush
        if not brush:
            self.report({'WARNING'}, "Aktiivista sivellintä ei löytynyt. Valitse Draw-sivellin.")
            return {'CANCELLED'}

        if InpaintState.active:
            InpaintState.active = False
            restore_brush_settings(context)
            self.report({'INFO'}, "OpenCV Inpaint -tila sammutettu.")
        else:
            primary_img, target_imgs = get_target_paint_images(context)
            if not primary_img or not target_imgs:
                self.report({'WARNING'}, "Valitse vähintään yksi Image Texture -node tai avaa kuva!")
                return {'CANCELLED'}

            if context.area.type == 'VIEW_3D':
                if context.mode != 'PAINT_TEXTURE':
                    bpy.ops.object.mode_set(mode='PAINT_TEXTURE')

                obj = context.active_object
                if obj and obj.active_material and obj.active_material.use_nodes:
                    nodes = obj.active_material.node_tree.nodes
                    for n in nodes:
                        if n.type == 'TEX_IMAGE' and n.image == primary_img:
                            nodes.active = n
                            break

            InpaintState.prev_color = tuple(brush.color)
            InpaintState.prev_blend = brush.blend
            InpaintState.prev_strength = brush.strength

            brush.color = (1.0, 0.0, 1.0)
            brush.blend = 'MIX'
            brush.strength = 1.0

            InpaintState.primary_image_name = primary_img.name
            save_snapshots(target_imgs)
            InpaintState.active = True

            # Käynnistetään modaalinen tarkkailija ajastimen sijaan
            bpy.ops.paint.cv_inpaint_watcher('INVOKE_DEFAULT')

            self.report({'INFO'}, f"Inpaint aktivoitu ({len(target_imgs)} tekstuuria). Maalaa ja vapauta hiiri.")

        for window in context.window_manager.windows:
            for area in window.screen.areas:
                area.tag_redraw()

        return {'FINISHED'}


class PAINT_OT_undo_cv_inpaint(bpy.types.Operator):
    bl_idname = "paint.undo_cv_inpaint"
    bl_label = "Kumoa edellinen Inpaint"

    def execute(self, context):
        if not InpaintState.undo_stack:
            self.report({'WARNING'}, "Ei kumottavaa Inpaint-toimintoa muistissa.")
            return {'CANCELLED'}

        for img_name, prev_arr in InpaintState.undo_stack.items():
            img = bpy.data.images.get(img_name)
            if img and (img.size[1], img.size[0], 4) == prev_arr.shape:
                img.pixels.foreach_set(prev_arr.ravel())
                img.update()
                InpaintState.snapshots_float[img_name] = prev_arr.copy()

        InpaintState.undo_stack.clear()
        self.report({'INFO'}, "Edellinen Inpaint kumottu kaikille tekstuureille.")
        return {'FINISHED'}


class PAINT_OT_refresh_cv_snapshot(bpy.types.Operator):
    bl_idname = "paint.refresh_cv_snapshot"
    bl_label = "Päivitä valitut nodet"

    def execute(self, context):
        primary_img, target_imgs = get_target_paint_images(context)
        if primary_img and target_imgs:
            InpaintState.primary_image_name = primary_img.name
            save_snapshots(target_imgs)
            self.report({'INFO'}, f"Päivitetty {len(target_imgs)} tekstuuria.")
        return {'FINISHED'}


def draw_inpaint_ui(self, context):
    layout = self.layout
    scene = context.scene

    if not HAS_CV2:
        box = layout.box()
        box.alert = True
        box.label(text="OpenCV (cv2) puuttuu!", icon='ERROR')
        return

    col = layout.column(align=True)
    if InpaintState.active:
        col.alert = True
        col.operator("paint.toggle_cv_inpaint", text="LOPETA INPAINT-TILA", icon='PAUSE')
        col.alert = False

        box = layout.box()
        box.label(text=f"Ohjaava: {InpaintState.primary_image_name}", icon='BRUSH_DATA')

        col_list = box.column(align=True)
        col_list.label(text=f"Kohteet ({len(InpaintState.target_image_names)} kpl):")
        for name in InpaintState.target_image_names:
            col_list.label(text=f"  • {name}", icon='IMAGE_DATA')

        row = box.row(align=True)
        row.operator("paint.undo_cv_inpaint", text="Kumoa", icon='LOOP_BACK')
        row.operator("paint.refresh_cv_snapshot", text="Päivitä", icon='FILE_REFRESH')
    else:
        col.operator("paint.toggle_cv_inpaint", text="Aloita OpenCV Inpaint", icon='PLAY')

        primary_img, target_imgs = get_target_paint_images(context)
        if target_imgs:
            box = layout.box()
            box.label(text=f"Valitut nodet ({len(target_imgs)} kpl):", icon='NODE_TEXTURE')
            for img in target_imgs:
                prefix = "[Canvas] " if img == primary_img else "• "
                box.label(text=f"  {prefix}{img.name}")

    layout.separator()
    col_settings = layout.column(align=True)
    col_settings.prop(scene, "cv_inpaint_radius", text="Inpaint Säde (px)")
    col_settings.prop(scene, "cv_inpaint_dilate", text="Maskin laajennus (px)")
    col_settings.prop(scene, "cv_inpaint_algo", text="Algoritmi")


class VIEW3D_PT_cv_inpaint(bpy.types.Panel):
    bl_label = "OpenCV Inpaint Brush"
    bl_idname = "VIEW3D_PT_cv_inpaint"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    def draw(self, context):
        draw_inpaint_ui(self, context)


class IMAGE_PT_cv_inpaint(bpy.types.Panel):
    bl_label = "OpenCV Inpaint Brush"
    bl_idname = "IMAGE_PT_cv_inpaint"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    def draw(self, context):
        draw_inpaint_ui(self, context)


classes = (
    PAINT_OT_cv_inpaint_watcher,
    PAINT_OT_toggle_cv_inpaint,
    PAINT_OT_undo_cv_inpaint,
    PAINT_OT_refresh_cv_snapshot,
    VIEW3D_PT_cv_inpaint,
    IMAGE_PT_cv_inpaint,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.cv_inpaint_radius = bpy.props.IntProperty(
        name="Inpaint Radius",
        default=5, min=1, max=100
    )
    bpy.types.Scene.cv_inpaint_dilate = bpy.props.IntProperty(
        name="Mask Dilate",
        default=2, min=0, max=25
    )
    bpy.types.Scene.cv_inpaint_algo = bpy.props.EnumProperty(
        name="Algorithm",
        items=[
            ('TELEA', "Telea (Fast Marching)", "Nopeampi ja tasaisempi pienille alueille"),
            ('NS', "Navier-Stokes", "Säilyttää paremmin jatkuvia linjoja"),
        ],
        default='TELEA'
    )


def unregister():
    InpaintState.active = False

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.cv_inpaint_radius
    del bpy.types.Scene.cv_inpaint_dilate
    del bpy.types.Scene.cv_inpaint_algo


if __name__ == "__main__":
    register()