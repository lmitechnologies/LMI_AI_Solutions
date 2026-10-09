## [2.0.0](https://github.com/lmitechnologies/LMI_AI_Solutions/compare/v1.6.0...v2.0.0) (2026-10-09)

### ⚠ BREAKING CHANGES

* **detectron2:** convert takes the engine size instead of a sample image (#425)
* **mask_utils:** quickhull, process and points_to_segments are removed from lmi_utils.postprocess_utils.mask_utils.
* **tiler:** Tiler.tile() no longer takes a mode and Tiler.untile() no longer takes scale_mode; pass scale_mode to the Tiler constructor instead. The second positional argument of untile() is now overlap_mode.
* **OD:** YOLO run_model.py is now infer.py, the detector CLIs take shared -w -i -o -c -s flags, and
--json replaces the csv output (csv eval scripts moved to eval_utils/deprecated/). postprocess() no longer
takes `operators`; predict() reverts the batch once. AnomalyDetector no longer takes tile_size/stride/tile_mode;
AD tiling goes through the pipeline's tile step.
* support gofactory 3 manifest schema. (#287)
* support batch inference for AD and decouple the tiler (#263)
* predict() now returns batched results — each value in
the output dict is a list of per-image arrays. Single-image callers must
index [0] to get the same data as before. Results.to_dict() parameter
renamed from return_tensor to return_numpy (inverted semantics).
Results.classes is now np.ndarray instead of list[str].
* split the to_mask into two functions and drop the 'mask_type' argument (#240)
* support yolo26 models and drop v0 support (#224)
* remove efficientnet, tf_objdet, and PaddleOCR submodules (#218)
* improve imports and stop importing from sub-directory paths (#207)
* - No longer support gadget version < 2.4
- Require pipeline to use global preprocessing

### Features

* add pip dependence bot ([#242](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/242)) ([ea8ad7f](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/ea8ad7f9c8ff1d36604193e130b5cf57d7b5282b))
* add pipeline_utils functions for subtractive masking / damping anomaly maps using instance segmentation ([#304](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/304)) ([db49983](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/db49983a57a0ac43b4ac0fc912f04f5513953d1b))
* **ad:** export dynamic-batch AD models and fix export edge cases ([#442](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/442)) ([0d2d1a9](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/0d2d1a901d7cdd88530209b8897db3ff6247ee31))
* **ad:** revert maps in predict when operators are given ([#440](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/440)) ([fa7b1c2](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/fa7b1c270c3d8225c306b33fd119f39a827b624c))
* anomalib v2.2.0 ([#182](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/182)) ([f3e45cd](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/f3e45cd12ee9c05bd8c00cc23d17f72f741943f5))
* **anomalib:** Implement tiler changes and max training sample estimation for AnomalibV2 ([#373](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/373)) ([92b05d6](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/92b05d6c05060069f16c1f7ae71f37d7219948ba))
* **anomalib:** upgrade to 2.6.0 ([#409](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/409)) ([857e361](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/857e361688ea824849a58beebed209bd87dc83eb))
* convert camelcase to snakecase for loading local models ([#249](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/249)) ([8276ecd](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/8276ecd58f2ca77ce7e26d18777fba256f8a4cfb))
* embed model metadata in onnx/engine files ([#382](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/382)) ([e02604d](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/e02604dcbae45d8678a272257900d652a4b431db))
* improve imports and stop importing from sub-directory paths ([#207](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/207)) ([1c3a8ab](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/1c3a8abee154034890a76da62aeea731443358c1))
* **od:** add rf-detr class-name sidecar support and fix inference bugs across backends ([#270](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/270)) ([78058d9](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/78058d987855d77a698ecb8fe8bdbe666d8dc1eb))
* **od:** draw the tile grid as semi-transparent dotted gray lines ([#413](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/413)) ([05f1000](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/05f1000eb4b40f267ad6e46dc6cd7e5dc1c8bac7))
* **od:** embed class names and color order in detectron2 exports ([#443](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/443)) ([3cf960a](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/3cf960a3fc078142227005aa4f74151f6340cbcb))
* **od:** run tiles in batches on dynamic-batch engines ([#412](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/412)) ([4b459bc](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/4b459bca7c9a8da7c524a2f29795fdc9fb2229a1))
* **OD:** support rf-detr training for gofactory ([#356](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/356)) ([800348b](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/800348b037eae9f19b8431e1a67189f1e8123677))
* **OD:** support static obb and pose models ([#402](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/402)) ([3c8e633](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/3c8e63328275023756c7db964990b6f89db6d162))
* **OD:** support tiling for OD models ([#401](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/401)) ([ed66c52](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/ed66c5225f812edefd416859c51fe8ae9ad242e7))
* **onnx:** take any batch on dynamic-batch models ([#438](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/438)) ([7617f05](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/7617f0571f0acb25b59a58ab4c16559d829f44b5))
* preprocessor supports both tiling and resize_and_pad ([#180](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/180)) ([4c2aa13](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/4c2aa13f70786c5e9046739ae97238fe76346368))
* **rfdetr:** infer model type from ckpt ([#384](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/384)) ([48443f8](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/48443f81e2e9dc5a9eccc40bf5e39d86a4cda3fb))
* **rotate:** Support rotation preprocess ([#403](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/403)) ([03f2439](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/03f243901e3f893be6d7199295656d0168bc010c))
* support ad onnx inference ([#291](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/291)) ([bcb45f8](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/bcb45f8c0988caf4af9870b54b084106b40c2d96))
* support batch inference for AD and decouple the tiler ([#263](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/263)) ([02c2dda](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/02c2ddabdb5caa95d1a73f1d93ac90e37557419b))
* support batch inference for classifiers ([#281](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/281)) ([8561a51](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/8561a51f6d9fba3faf6d9da57f3edf8ecb68569f))
* support flipping labels ([#190](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/190)) ([fb5c4be](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/fb5c4bec5d0bc037f1756cf294e9ea2be61a0016))
* support gofactory 3 manifest schema. ([#287](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/287)) ([5cfc208](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/5cfc2084ad33c708ff29f3d37961632ac483a230))
* support json to factory format conversion ([#231](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/231)) ([6df6d4c](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/6df6d4c829d9cdd45c1204dd29bfffbd29d4e07b))
* support tensors and refactor model register ([#262](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/262)) ([ca2c637](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/ca2c63715b9d4de40733fd1ed5bb186d1033f801))
* support torchscript extension ([#288](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/288)) ([7b37938](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/7b379381ba1c6aa00824e0f7cfb6e3b65dcda9c0))
* support yolo26 models and drop v0 support ([#224](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/224)) ([6acf96b](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/6acf96b39e8ab9dc1887c7329041858e50c1365e))
* supporting torchscript for anomalib v2 ([#192](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/192)) ([9278215](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/9278215a3e92791ce1a7226643a834bb1eb9f5a6))
* update rf-detr to 1.4.1 ([#191](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/191)) ([90ae0c3](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/90ae0c32ffdf53454891370936d1412c999ed5a0))

### Bug Fixes

* **AD:** auto resize to input shape when call revert_preprocess ([#319](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/319)) ([fbaf6ec](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/fbaf6ec47a257921977976a8b57857fa6e49a802))
* add default resize for OD models ([#305](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/305)) ([35b1f85](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/35b1f8559c66e43f5acaebaa4dc28a40929b4e59))
* add extra resize if input sizes mismatch with model's expected size ([#307](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/307)) ([1ba3d1d](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/1ba3d1dca07daa849fc0cdccd9ececbb87196a03))
* add padding value for default resize ([#308](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/308)) ([b4fe18e](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/b4fe18eaedccff7f1272ce0585e2b6045f9f6319))
* add required metadata for yolo converting to engines ([#372](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/372)) ([f7af4af](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/f7af4afb892a4ecfa6dfd1a3c05dabd9940f30e4))
* Added OrientedObjectDetection to schemav3 ([#400](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/400)) ([4eb295d](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/4eb295d5e9ada8ff34e46caf5c0f814b54cff08e))
* **ad:** drop tile arguments from AnomalyDetector construction ([#404](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/404)) ([e46e8d6](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/e46e8d60079956805037ce5ae574674898da80b6))
* align pose keypoint slots and preserve visibility ([858c8e4](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/858c8e44abdf78245ecdb29316a90bbd22628be0))
* **anomalib:** export and convert tiled models from any device ([#415](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/415)) ([ea1aaaa](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/ea1aaaa4094d770e93a38daf37949b3690fdfff8))
* **anomalib:** fix the 2.6.0 upgrade's deps, ONNX export and float16 handling ([#414](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/414)) ([6b87197](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/6b87197e0312a42158d649db90fe24665b8260a5))
* **bbox:** lossless rotated-box conversion ([#366](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/366)) ([5a069ce](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/5a069ce8b36fac1089d23df3ddabea4bc32c185d))
* bug fixes for registering models ([#340](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/340)) ([216d0e3](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/216d0e3b426ce5f441479d1fbfe0ba30121fec85))
* correct OD resize-injection size check and empty-batch coordinate reconstruction ([#346](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/346)) ([22556c8](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/22556c862b990a6f6fe5789017ff94031ca51421))
* **deps:** scope the uv lock to Linux so Dependabot can resolve again ([#429](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/429)) ([6d657f6](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/6d657f66b2f39cba810662d2c1dc04e01b3bf80c))
* detectron2 conversion update. ([#195](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/195)) ([17197ab](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/17197abba776c34e837a9fa0822d3641ee7817ed))
* **detectron2:** build training on detectron2's own mapper and config merge ([#419](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/419)) ([76e3c63](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/76e3c634f02524dd1e6d8a4912c40b11064a0d49))
* **detectron2:** match the original model in converted TRT engines ([#417](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/417)) ([490e2a9](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/490e2a9c11cd303a15038790ffd49d91f59c0adb))
* deterministic model resource teardown in pipeline clean_up ([#347](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/347)) ([398238d](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/398238da9fe1d843fd3b41151d426040df6f3a36))
* **engines:** load TRT on the requested GPU, support TRT 8.5, harden input and device checks ([#435](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/435)) ([1203029](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/120302952e7a5bc191bf71add77b58eb3f2751eb))
* **engines:** reject TRT outputs not sized by batch, tighten ONNX buffers and sync ([#437](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/437)) ([f66c212](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/f66c212aeb3d08cb9b63eb195c954c99dc76822f))
* ensure ad_max > threshold to prevent weird heatmap ([#179](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/179)) ([ce7dcde](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/ce7dcde9b6a63d766364d61b3a495aff6c1fcf3c))
* **export:** convert shapes to what each target format can express ([#367](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/367)) ([e757c52](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/e757c52ab32ac5f16c30368d9a1acfd17863b337))
* **file_utils:** fix the dim reads from images ([ab38787](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/ab387878a070605dc9f2e972dddea9b3e70cfaaa))
* fix an dependency error when install anomalib v1.1.1 ([#229](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/229)) ([beb3df1](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/beb3df1671e087e66223756ac23face79d0e6b22))
* fix bugs in lst_to_json ([#236](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/236)) ([8a9cc95](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/8a9cc9534404d23c728deb307071eba59a35e481))
* fix bugs in representations ([#238](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/238)) ([81f0ad8](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/81f0ad88961333a8475ebb3fbf68bac437a03844))
* fix CI errors when import albumentations and albucore ([#199](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/199)) ([ebda9f3](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/ebda9f3947bb1ef2a1dd1ddef480a1a14b30f83e))
* fix local unit tests ([#184](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/184)) ([d7afcd3](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/d7afcd3ac5a7e237fbd526841b8f88431b46c2de))
* fix missing base commit ([6205662](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/620566269cb6c1e8ff81577c57bbb20d62e52fab))
* fix preprocess bug for detectron2 ([#275](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/275)) ([153485a](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/153485aa150c58d8b1863cb09820e5abd29f26a4))
* fix the bugs in json_to_coco script ([#230](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/230)) ([09ceeda](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/09ceeda363a3d14f453e60165502ebd62ae8f148))
* fix the dependencies with rfdetr ([#239](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/239)) ([53079b4](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/53079b4c020afe2c6c4f59480216ab272caeec38))
* fix the false trigger for precommit updates ([#232](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/232)) ([ff790e0](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/ff790e0ac8bcf9c583f4079d7194c0ce7b31a0bf))
* fix the race condition where the CI runs before finishing building images ([#260](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/260)) ([108a242](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/108a2429857d5e1d5e16c468f9fa3403173c7cb8))
* **GoFactory integration:** Changes needed updating GoFactory to latest ([#326](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/326)) ([ea74e31](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/ea74e316770143d908aa448f3f5a121ba8f6685d))
* harden pipeline model loading and prediction result handling ([306d9f8](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/306d9f811a2268f073b23b642a03f1e6ea9ab7bb))
* lazy backend discovery and better registry error reporting ([#341](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/341)) ([26a041e](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/26a041e713e12a7fa3d628d045f9bd1d4cd0ffa6))
* **od:** require one history entry per image in predict ([#441](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/441)) ([2cc91b5](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/2cc91b5fd56bfaa92bca0ceca04e39bdee441093))
* **od:** trace tiled segments from the merged masks ([#410](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/410)) ([dd564d6](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/dd564d67b04c818df91ef979440c8f1af50f8106))
* **onnx:** wait for torch to finish writing inputs before ORT reads them ([#426](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/426)) ([44517cf](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/44517cfee433e2c73c0b4deeed3f06fd0cc50436))
* **pipeline:** match revert_preprocess to predict; keep OD coords as floats clamped at 0 ([#439](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/439)) ([af6a66a](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/af6a66ab4b623ea684e686b9253385d8939e890d))
* **resize:** match dataset resizes to inference and train AD v2 with antialias off ([#427](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/427)) ([a0b2657](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/a0b2657079115079ffd9a4b231b73efe0c964f54))
* **resize:** match inference resizes to training resizes ([#408](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/408)) ([19c15e7](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/19c15e7b2ba651b59ef4c00dd0bc024b4c2784f6))
* **rfdetr:** support onnx backend, batch_size arg, and sparse COCO class ids ([#381](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/381)) ([134f1c2](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/134f1c21161762c3d96a532cfe4d052c6e744c19))
* static manifest parsing for running pipeline unit-test ([#297](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/297)) ([512da1c](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/512da1caea7d6dfcd5a84a74b7eb4e2f95951eea))
* strip right/bottom-only pad in mask revert, batch AD inverse-resize, harden update_results ([792a525](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/792a52595ba0231c7d22b92d573ce94f0b8499df))
* **test:** fix numpy version mismatch issue ([7a07082](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/7a07082a3ed23ac4012d956bf3b9fe0147c74922))
* **tile:** merge seam gaps ([#407](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/407)) ([57b09c4](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/57b09c404f3047cf3e83080244cfea0c65360bfc))
* **tiler:** bind the scale mode to the tiler and fix untiling ([#406](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/406)) ([3096b89](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/3096b89b1549c8cbcd8dc3055cbcead100ff4830))
* **tiling:** fix tiling bug calculating distances ([#363](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/363)) ([23830f1](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/23830f1941d410f46698d63e883e612821fdde06))
* use class map for val ([e15fa8e](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/e15fa8eb5dd6c4030a3ec798956f0edad1fcd507))
* **yolo:** correctly infer the imgsz from weights files ([#368](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/368)) ([ec3756f](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/ec3756f531aac1b0591294dd185b901a8181096d))

### Performance Improvements

* **json-to-yolo:** improve the speed of the script ([fcbe503](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/fcbe50302e49ca7d5d85f7942f08d13a4b81b255))

### Miscellaneous Chores

* remove efficientnet, tf_objdet, and PaddleOCR submodules ([#218](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/218)) ([46b3877](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/46b387720c88a765e76dbe19be53495bc79069e4))

### Code Refactoring

* consolidate inference pipeline into ODBase with batch support ([#250](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/250)) ([ee0b41d](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/ee0b41d55ca0620e3a7f64312a799ae137eea4d8))
* **detectron2:** convert takes the engine size instead of a sample image ([#425](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/425)) ([d207f15](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/d207f152b6eab5175462fc8599bbd0f0f4fb95d7))
* **mask_utils:** replace the numba convex hull with OpenCV ([#422](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/422)) ([3f16517](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/3f16517485ea0169cec48b39ffec6616e459ae77))
* split the to_mask into two functions and drop the 'mask_type' argument ([#240](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/240)) ([b758071](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/b758071b6cea9bd40f1fb8f85047de44b6964ff3))

# [1.6.0](https://github.com/lmitechnologies/LMI_AI_Solutions/compare/v1.5.2...v1.6.0) (2026-01-20)


### Bug Fixes

* auto update release versions in pyproject.toml  ([#178](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/178)) ([1d9db22](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/1d9db229e2473eb9e08b47ec12fe55eb9be7b592))
* lower ultralytics version to 8.3 ([#169](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/169)) ([f5810ce](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/f5810ce1139409f9d751b7561dc2ba2ae9405c13))


### Features

* rfdetr ([e899fa4](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/e899fa436e37af4cb7a382ca3e0c050839d32750))

## [1.5.2](https://github.com/lmitechnologies/LMI_AI_Solutions/compare/v1.5.1...v1.5.2) (2026-01-09)


### Bug Fixes

* fix trt inference for ad models ([#167](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/167)) ([2c79f9c](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/2c79f9ca128b06ab83872ac457380676b280c7db))

## [1.5.1](https://github.com/lmitechnologies/LMI_AI_Solutions/compare/v1.5.0...v1.5.1) (2025-12-29)


### Bug Fixes

* model fuse on ARM ([#163](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/163)) ([b4598a5](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/b4598a54c46b5fdb9f2b59e5befea0e00fba9dfd))

# [1.5.0](https://github.com/lmitechnologies/LMI_AI_Solutions/compare/v1.4.0...v1.5.0) (2025-12-17)


### Features

* Support running torch trace AD models on CPU ([#159](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/159)) ([b2dfc66](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/b2dfc66536ee96fc46d545fc2f5e56bc49eed304))

# [1.4.0](https://github.com/lmitechnologies/LMI_AI_Solutions/compare/v1.3.1...v1.4.0) (2025-11-25)


### Features

* update pipeline base class to 2.4.145 ([d2e78ac](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/d2e78ac524daab9d44392c758c239ff15633fb9d))


### Reverts

* Revert "upgrade pipeline base class to 2.4.145 ([#156](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/156))" ([dca47f6](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/dca47f61348f85ea1acae39bf8ad70d6238795ab))

## [1.3.1](https://github.com/lmitechnologies/LMI_AI_Solutions/compare/v1.3.0...v1.3.1) (2025-11-25)


### Reverts

* Revert "update base class to match with gadget version 2.4.145 ([#155](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/155))" ([c352a4c](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/c352a4c2b09f5b2e86c11c574bbd2aa6fd0ecc21))

# [1.3.0](https://github.com/lmitechnologies/LMI_AI_Solutions/compare/v1.2.0...v1.3.0) (2025-11-25)


### Features

* tilling initialization for ad ([aee673c](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/aee673c6518e0d3e58fb6b0096cc3066af7ce268))

# [1.2.0](https://github.com/lmitechnologies/LMI_AI_Solutions/compare/v1.1.1...v1.2.0) (2025-11-12)


### Bug Fixes

* fix the bug where run_model does not load correct yolo seg models ([#152](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/152)) ([ad83e05](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/ad83e051b23774e97696ce28f350aa1a4b48680d))


### Features

* Preprocessor module for automatic preprocessing ([402da43](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/402da43e26d1d998e2206d2faaa68d4797330d22))

## [1.1.1](https://github.com/lmitechnologies/LMI_AI_Solutions/compare/v1.1.0...v1.1.1) (2025-10-30)


### Bug Fixes

* fixed image_size issue with classifiers ([#149](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/149)) ([a2b621a](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/a2b621a3e7b9fd607830e2d38a60f9fcc3c85fe3))

# [1.1.0](https://github.com/lmitechnologies/LMI_AI_Solutions/compare/v1.0.0...v1.1.0) (2025-10-28)


### Features

* adding pipeline_base to the AIS repo ([#145](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/145)) ([567facf](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/567facf2aed3bd3808f028593ef3e20e21d82937))

# 1.0.0 (2025-10-28)


### Features

* add release bot ([#148](https://github.com/lmitechnologies/LMI_AI_Solutions/issues/148)) ([734fc9f](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/734fc9fe14b031a78595c6c05852b31fc63a8b09))


### Reverts

* Revert "Updated yolov5 READMe.md" ([ecf3e15](https://github.com/lmitechnologies/LMI_AI_Solutions/commit/ecf3e15c776b04033e17a58b7a788742f16e2edd))
