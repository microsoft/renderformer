# Teaser Scenes
## Cornell Box
python3 scene_processor/convert_scene.py examples/cbox.json --output_h5_path output/cbox/cbox.h5
python3 infer.py --h5_file output/cbox/cbox.h5

## Tree
python3 scene_processor/convert_scene.py examples/tree.json --output_h5_path output/tree/tree.h5
python3 infer.py --h5_file output/tree/tree.h5

## Constant Width Shapes
python3 scene_processor/convert_scene.py examples/constant-width.json --output_h5_path output/constant-width/constant-width.h5
python3 infer.py --h5_file output/constant-width/constant-width.h5

## Compose Scene
python3 scene_processor/convert_scene.py examples/compose-scene.json --output_h5_path output/compose-scene/compose-scene.h5
python3 infer.py --h5_file output/compose-scene/compose-scene.h5

# Other Static Scenes
## Renderformer Logo
python3 scene_processor/convert_scene.py examples/renderformer-logo.json --output_h5_path output/renderformer-logo/renderformer-logo.h5
python3 infer.py --h5_file output/renderformer-logo/renderformer-logo.h5

## Cornell Box Lucy
python3 scene_processor/convert_scene.py examples/cbox-lucy.json --output_h5_path output/cbox-lucy/cbox-lucy.h5
python3 infer.py --h5_file output/cbox-lucy/cbox-lucy.h5

## Cornell Box Bunny
python3 scene_processor/convert_scene.py examples/cbox-bunny.json --output_h5_path output/cbox-bunny/cbox-bunny.h5
python3 infer.py --h5_file output/cbox-bunny/cbox-bunny.h5

## Cornell Box Teapot
python3 scene_processor/convert_scene.py examples/cbox-teapot.json --output_h5_path output/cbox-teapot/cbox-teapot.h5
python3 infer.py --h5_file output/cbox-teapot/cbox-teapot.h5

## Shader Ball
python3 scene_processor/convert_scene.py examples/shader-ball.json --output_h5_path output/shader-ball/shader-ball.h5
python3 infer.py --h5_file output/shader-ball/shader-ball.h5 --tone_mapper agx

## Veach MIS
python3 scene_processor/convert_scene.py examples/veach-mis.json --output_h5_path output/veach-mis/veach-mis.h5
python3 infer.py --h5_file output/veach-mis/veach-mis.h5 --tone_mapper pbr_neutral

## Room
python3 scene_processor/convert_scene.py examples/room.json --output_h5_path output/room/room.h5
python3 infer.py --h5_file output/room/room.h5 --tone_mapper agx

## Horse and Heart
python3 scene_processor/convert_scene.py examples/horse-and-heart.json --output_h5_path output/horse-and-heart/horse-and-heart.h5
python3 infer.py --h5_file output/horse-and-heart/horse-and-heart.h5 --tone_mapper agx

## Fox in the Wild
python3 scene_processor/convert_scene.py examples/fox-in-the-wild.json --output_h5_path output/fox-in-the-wild/fox-in-the-wild.h5
python3 infer.py --h5_file output/fox-in-the-wild/fox-in-the-wild.h5 --tone_mapper agx

## Crystals
python3 scene_processor/convert_scene.py examples/crystals.json --output_h5_path output/crystals/crystals.h5
python3 infer.py --h5_file output/crystals/crystals.h5 --tone_mapper agx
