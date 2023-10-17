<script lang="ts">
  import {
    AbstractMesh,
    ArcRotateCamera, Axis, Color3,
    Engine,
    HemisphericLight,
    ImageProcessingConfiguration,
    Mesh,
    PBRMetallicRoughnessMaterial,
    Scene,
    SceneLoader, Space,
    Vector3
  } from "@babylonjs/core";
  import { MTLFileLoader, OBJFileLoader } from "@babylonjs/loaders/OBJ";
  import { Inspector } from "@babylonjs/inspector";
  import { onMount } from "svelte";
  import JSZip from "jszip";
  import RangeSlider from "svelte-range-slider-pips";
  import { PlanarRange } from "planar-range";

  let canvas: HTMLCanvasElement;

  let includeTexture = true;

  let currentTab = "ez";

  let currentImgHash: string | null = null;

  let happyValue = 0;
  let sadValue = 0;
  let fearValue = 0;
  let angerValue = 0;
  let surpriseValue = 0;
  let disgustValue = 0;

  let expArr: number[] = Array(50).fill(0);
  let poseArr: number[] = Array(6).fill(0);
  const poseArrLabels = ["Not used x", "Not used y", "Not used z", "Jaw up/down", "Jaw left/right", "Jaw twist left/right"];
  let neckPoseArr: number[] = Array(3).fill(0);
  const neckPoseArrLabels = ["Neck up/down", "Neck left/right", "Neck twist left/right"];
  let eyePoseArr: number[] = Array(6).fill(0);

  let onSelectUploadFile: (e: Event & { currentTarget: (EventTarget & HTMLInputElement) }) => void | null = null;
  let applyEmotionFunc;
  let applyManualValuesFunc;

  let showManualTab = false;
  let isProcessing = false;

  let mouthOpenness = [0];
  const mountOpennessInfo = {
    0: { label: "Closed", value: 0 },
    1: { label: "Slightly open", value: 0.15 },
    2: { label: "Open", value: 0.25 },
    3: { label: "Wide open", value: 0.35 }
  };

  $: if (currentTab === "ez") poseArr[3] = mountOpennessInfo[mouthOpenness[0]].value;

  let eyebrowsRaise = [0];
  const eyebrowsRaiseInfo = {
    "-1": { label: "Lowered", value: -2 },
    0: { label: "Relaxed", value: 0 },
    1: { label: "Slightly raised", value: 2 },
    2: { label: "Raised", value: 4 }
  };
  $: if (currentTab === "ez") expArr[9] = eyebrowsRaiseInfo[eyebrowsRaise[0]].value;

  let lookDirection = [0.5, 0.5];
  $ : if (currentTab === "ez") {
    neckPoseArr[0] = (lookDirection[1] * 2 - 1) * 0.7;
    neckPoseArr[1] = (lookDirection[0] * 2 - 1) * 0.7;
  }

  let showFaceTexture = true;

  let headNode: AbstractMesh | null;

  $:if (headNode) {
    headNode.useVertexColors = showFaceTexture;
  }

  const isProduction = import.meta.env.PROD;
  const backendUrl = isProduction ? import.meta.env.BASE_URL : "http://localhost:8000/";

  let currentMeshZipBlob: Blob | null = null;

  let hairMaterial: PBRMetallicRoughnessMaterial | null = null;

  const hexColorToRGBNormalized = (hexColor: string) => {
    const hex = hexColor.replace('#', '');
    const r = parseInt(hex.substring(0, 2), 16) / 255;
    const g = parseInt(hex.substring(2, 4), 16) / 255;
    const b = parseInt(hex.substring(4, 6), 16) / 255;
    return new Color3(r, g, b);
  }

  let hairColor = '#858585';
  $: if (hairMaterial) {
    hairMaterial._albedoColor = hexColorToRGBNormalized(hairColor);
  }


  onMount(async () => {
    OBJFileLoader.COMPUTE_NORMALS = true;
    OBJFileLoader.OPTIMIZE_WITH_UV = true;
    OBJFileLoader.OPTIMIZE_NORMALS = true;
    OBJFileLoader.IMPORT_VERTEX_COLORS = true;

    const engine = new Engine(canvas, true);

    const scene = new Scene(engine);
    scene.imageProcessingConfiguration.toneMappingEnabled = true;
    scene.imageProcessingConfiguration.toneMappingType = ImageProcessingConfiguration.TONEMAPPING_ACES;
    scene.imageProcessingConfiguration.contrast = 1.9;

    scene.useRightHandedSystem = true;

    const rootNode = new Mesh("root", scene);

    const camera = new ArcRotateCamera(
      "camera",
      Math.PI / 2,
      Math.PI / 2.5,
      0.5,
      new Vector3(0, 0, 0)
    );
    camera.minZ = 0.1;
    camera.attachControl(canvas, true);
    camera.wheelPrecision = 300;
    camera.pinchPrecision = 500;
    camera.panningSensibility = 1000;
    camera.panningInertia = 0.3;

    if (isProduction) {
      camera.lowerRadiusLimit = 0.4;
      camera.upperRadiusLimit = 1.2;
    }


    // the variable isn't used, but it will make the light appear in the scene
    const light = new HemisphericLight(
      "HemiLight",
      new Vector3(0, 0.2, .8),
      scene
    );

    const headMaterial = new PBRMetallicRoughnessMaterial("headMaterial", scene);
    headMaterial.backFaceCulling = false;
    headMaterial.metallic = 0.0;
    headMaterial.roughness = 0.6;

    hairMaterial = new PBRMetallicRoughnessMaterial("hairMaterial", scene);
    hairMaterial.metallic = 0.0;
    hairMaterial.roughness = 0.3;
    hairMaterial._albedoColor = new Color3(1.0, 0.8, 0.6);

    engine.runRenderLoop(function() {
      scene.render();
    });

    window.addEventListener("resize", function() {
      engine.resize();
    });

    // Drag n drop upload
    document.body.addEventListener("dragover", (ev) => {
        // Prevent default behavior (Prevent file from being opened)
        ev.preventDefault();
      }
    );

    document.body.addEventListener("drop", (ev) => {
      // Prevent default behavior (Prevent file from being opened)
      ev.preventDefault();

      let file;

      if (ev.dataTransfer.items?.length > 0) {
        const firstItem = ev.dataTransfer.items[0];
        if (firstItem.kind === "file") {
          file = firstItem.getAsFile();
        }
      } else if (ev.dataTransfer.files?.length > 0) {
        file = ev.dataTransfer.files[0];
      }

      if (file != null) {
        processImage(file);
      }
    });

    function loadDevFeatures() {
      if (!isSmallScreenDevice()) {
        Inspector.Show(scene, {});
      }
      showManualTab = true;
    }

    document.body.addEventListener("click", nClicks(3, 400)(loadDevFeatures));
    if (!isProduction) {
      loadDevFeatures();
    }

    // Click upload
    onSelectUploadFile = async (e) => {
      const file = e.currentTarget.files[0];
      if (file != null) {
        processImage(file);
      }
      e.currentTarget.value = null;
    };

    applyEmotionFunc = () => {
      const emotionArr = [
        happyValue,
        sadValue,
        fearValue,
        angerValue,
        surpriseValue,
        disgustValue
      ];

      if (currentImgHash != null) {
        updateFlameMesh(currentImgHash, { emotionArr });
      }
    };

    applyManualValuesFunc = () => {
      if (currentImgHash != null) {
        updateFlameMesh(currentImgHash, { expArr, poseArr, eyePoseArr, neckPoseArr });
      }
    };

    function disposeScene() {
      for (const child of rootNode.getChildren()) {
        child.dispose();
      }
    }

    const loadFile = function(url) {
      disposeScene();

      if (url != null) {
        const mesh = SceneLoader.ImportMeshAsync(
          "",
          url,
          undefined,
          undefined,
          undefined,
          ".obj"
        );
        mesh.then((result) => {
          result.meshes.forEach((mesh) => {
            mesh.parent = rootNode;
          });
        });
      }
    };

    async function loadFileByName(url, name) {
      if (url == null) {
        return;
      }
      if (name.endsWith("detail.obj")) {
        SceneLoader.ImportMeshAsync(
          "",
          url,
          undefined,
          undefined,
          undefined,
          ".obj"
        ).then((result) => {
          result.meshes.forEach((mesh) => {
            headNode = mesh;
            mesh.name = "head";
            mesh.parent = rootNode;
            mesh.material = headMaterial;
          });
        });
      } else if (name.endsWith(".mtl")) {
        const blob = await fetch(url).then((response) => response.text());
        new MTLFileLoader().parseMTL(
          scene,
          blob,
          url,
          null
        );
      } else if (name.endsWith("normals.png")) {

      } else if (name.endsWith(".png")) {
        // headMaterial._albedoTexture = new Texture(url, scene, undefined, false);
      } else if (name === "metadata.json") {
        const metadata = await fetch(url).then((response) => response.json());

        const jawUpper = scene.getNodeByName("jaw_up") as Mesh | null;
        const jawLower = scene.getNodeByName("jaw_down") as Mesh | null;
        if (jawUpper && jawLower) {
          jawUpper.position = new Vector3(...metadata.lips.upper.center);
          jawLower.position = new Vector3(...metadata.lips.lower.center);

          jawUpper.translate(Axis.Z, -0.025, Space.LOCAL);
          jawLower.translate(Axis.Z, -0.025, Space.LOCAL);
          jawUpper.translate(Axis.Y, -0.015, Space.LOCAL);
          jawLower.translate(Axis.Y, 0.015, Space.LOCAL);

          jawUpper.rotation = new Vector3(0, -Math.PI / 2, 0);
          jawLower.rotation = new Vector3(0, -Math.PI / 2, 0);

          jawLower.scaling = jawUpper.scaling = new Vector3(0.01, 0.01, 0.014);
        }

        const hair = scene.getNodeByName("hair") as Mesh | null;
        if (hair) {
          hair.position = new Vector3(...metadata.hair_translation);
        }
        const hairContainer = scene.getNodeByName("hair_container") as Mesh | null;
        if (hairContainer) {
          hairContainer.scaling = new Vector3(...metadata.hair_scale);
        }

        console.log(metadata);
      } else if (name === "teeth.glb") {
        // const result = await SceneLoader.ImportMeshAsync(
        //   "",
        //   url,
        //   undefined,
        //   undefined,
        //   undefined,
        //   ".glb"
        // );
        // result.meshes.forEach((mesh) => {
        //   mesh.parent = rootNode;
        // });
      } else if (name === "hair.obj") {
        const result = await SceneLoader.ImportMeshAsync(
          "",
          url,
          undefined,
          undefined,
          undefined,
          ".obj"
        );
        const hairContainer = new Mesh("hair_container", scene);
        hairContainer.parent = rootNode;

        result.meshes.forEach((mesh) => {
          mesh.parent = hairContainer;
          mesh.name = "hair";
          mesh.visibility = 0.9;
          mesh.material = hairMaterial;
        });
      }
    }

    async function processZipBlob(blob: Blob) {
      const zip = new JSZip();

      const deferredFunctions = [];
      const zipPromises = [];
      const zipFiles = await zip.loadAsync(blob);

      zipFiles.forEach((relativePath, zipEntry) => {
        console.log(relativePath);
        const result = zipEntry.async("blob").then((blob) => {
          const blobUrl = URL.createObjectURL(blob);
          if (relativePath === "metadata.json") {
            deferredFunctions.push(() => loadFileByName(blobUrl, relativePath));
          } else {
            loadFileByName(blobUrl, relativePath);
          }
        });
        zipPromises.push(result);
      });

      await Promise.all(zipPromises);

      for await (const func of deferredFunctions) {
        await func();
      }
    }

    async function processImage(imageFile) {
      if (isProcessing) {
        return;
      }
      isProcessing = true;
      const formData = new FormData();
      formData.append("file", imageFile);

      const blob = await fetch(//http://192.168.1.29:8001/ or http://localhost:8000/
        `${backendUrl}upload?include_tex=${includeTexture}`, {
          method: "POST",
          body: formData
        }).then((response) => {
        currentImgHash = response.headers.get("Image-Hash");
        return response.blob();
      });

      currentMeshZipBlob = blob;

      disposeScene();

      await processZipBlob(blob);

      isProcessing = false;
    }

    // flameModifiers is an object which can take:
    // emotionArr : [0,1]x6
    // expArr: [-inf, inf]x50
    // poseArr: [-inf, inf]x6
    // neckPoseArr: [-inf, inf]x3
    // eyePoseArr: [-inf, inf]x6
    async function updateFlameMesh(imgHash, flameModifiers) {
      if (isProcessing) {
        return;
      }
      isProcessing = true;

      const requestOptions = {
        method: "POST",
        body: JSON.stringify({
          imgHash,
          includeTex: includeTexture,
          ...flameModifiers
        })
      };

      const blob = await fetch(
        `${backendUrl}update`,
        requestOptions
      ).then((response) => {
        return response.blob();
      });

      currentMeshZipBlob = blob;

      disposeScene();

      await processZipBlob(blob);

      isProcessing = false;
    }
  });

  function resetConfigParams() {
    happyValue = 0;
    sadValue = 0;
    fearValue = 0;
    angerValue = 0;
    surpriseValue = 0;
    disgustValue = 0;

    expArr = Array(50).fill(0);
    poseArr = Array(6).fill(0);
    neckPoseArr = Array(3).fill(0);
    eyePoseArr = Array(6).fill(0);

    mouthOpenness = [0];
    eyebrowsRaise = [0];
    lookDirection = [0.5, 0.5];
  }

  function isSmallScreenDevice() {
    return window.innerWidth < 960;
  }

  const nClicks = (minClickStreak, maxClickInterval = 500, resetImmediately = true) => {
    let timerId = 0;
    let clickCount = 0;
    let lastTarget = null;
    const reset = () => {
      timerId = 0;
      clickCount = 0;
      lastTarget = null;
    };

    return (originalEventHandler) => (e) => {
      if (lastTarget == null || lastTarget == e.target) { // 2. unless we clicked same target
        clickCount++; // 3. then increment click count
        clearTimeout(timerId);
      }
      lastTarget = e.target;
      timerId = setTimeout(reset, maxClickInterval); // 1. reset state within set time
      if (clickCount >= minClickStreak) {
        originalEventHandler(e);
        if (resetImmediately) {
          clickCount = 0;
        }
      }
    };
  };

  // Scroll to bottom when first loaded
  window.scrollTo(0, document.body.scrollHeight);
</script>

<main class="dark:text-slate-200">
  <div class="flex flex-col md:flex-row">
    <div class="flex-1 fixed md:relative top-0 left-0 max-h-[100vh]">
      <!-- download button  -->
      {#if currentMeshZipBlob != null}
        <div class="absolute top-3 left-1/2 mx-auto z-10 translate-x-[-6rem]">
          <a href={URL.createObjectURL(currentMeshZipBlob)}
             class="bg-blue-500/20 backdrop-blur-lg hover:bg-blue-500/50 ease-in-out duration-150 text-slate-100 hover:text-slate-100 w-[12rem] block text-lg p-2 rounded text-center cursor-pointer"
             download="model.zip">Download model</a>
        </div>
      {/if}
      <div class="test max-h-[100vh]">
        <canvas bind:this={canvas} class="touch-none w-[100vw] h-[80vh] z-10 md:w-full md:h-[100vh]"
                on:wheel|preventDefault />
      </div>
    </div>
    <div
      class="relative px-4 w-full rounded-t-xl mt-[77vh] overscroll-none bg-white pb-4 md:mt-0 md:h-[100vh] md:overflow-auto md:w-[25rem] md:rounded-t-none dark:bg-gray-700">
      {#if isProcessing}
        <div
          class="absolute top-0 bottom-0 left-0 right-0 bg-blue-50/50 z-20 backdrop-blur-sm flex gap-2 items-center justify-center">
          <div
            class="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-current border-r-transparent align-[-0.125em] motion-reduce:animate-[spin_1.5s_linear_infinite]"
            role="status">
          </div>
          <div>Processing...</div>
        </div>
      {/if}
      <div class="sticky z-10 pt-4 top-0 bg-white dark:bg-gray-700">
        <label for="myFile"
               class=" block bg-blue-500 hover:bg-blue-700 text-slate-100 font-bold text-2xl mb-3 py-5 rounded left-0 right-0 text-center cursor-pointer">
          Upload your portrait...</label>
        <input class="hidden" type="file" id="myFile" accept="image/*" on:change={onSelectUploadFile} />

        <div class="flex items-center my-2">
          <input id="checked-checkbox" type="checkbox" bind:checked={includeTexture}
                 class="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 focus:ring-2">
          <label for="checked-checkbox"
                 class="ml-2 text-sm font-medium text-gray-900 dark:text-slate-200 cursor-pointer">Create a 3D
            face texture with your portrait</label>
        </div>

        {#if currentImgHash != null }
          <div
            class="text-sm font-medium text-center text-gray-500 border-b border-gray-200 mb-2 -mx-4">
            <ul class="flex flex-wrap -mb-px">
              <li class="mr-2">
                <a href="#ez"
                   on:click|preventDefault={() => currentTab = 'ez'}
                   class="{
           currentTab === 'ez'
           ? 'inline-block p-4 text-blue-600 border-b-2 border-blue-600 rounded-t-lg active dark:text-slate-200 dark:border-blue-600'
           : 'inline-block p-4 border-b-2 border-transparent rounded-t-lg hover:text-gray-600 hover:border-gray-300 dark:hover:text-gray-300 dark:text-gray-400'}"
                >EZ Mode</a>
              </li>
              <li class="mr-2">
                <a href="#emotions"
                   on:click|preventDefault={() => currentTab = 'emotions'}
                   class="{
           currentTab === 'emotions'
           ? 'inline-block p-4 text-blue-600 border-b-2 border-blue-600 rounded-t-lg active dark:text-slate-200 dark:border-blue-600'
           : 'inline-block p-4 border-b-2 border-transparent rounded-t-lg hover:text-gray-600 hover:border-gray-300 dark:hover:text-gray-300 dark:text-gray-400'}">Emotions</a>
              </li>
              {#if showManualTab}
                <li class="mr-2">
                  <a href="#manual"
                     on:click|preventDefault={() => currentTab = 'manual'}
                     class="{
           currentTab === 'manual'
           ? 'inline-block p-4 text-blue-600 border-b-2 border-blue-600 rounded-t-lg active dark:text-slate-200 dark:border-blue-600'
           : 'inline-block p-4 border-b-2 border-transparent rounded-t-lg hover:text-gray-600 hover:border-gray-300 dark:hover:text-gray-300 dark:text-gray-400'}"
                  >Manual</a>
                </li>
              {/if}
            </ul>
          </div>
          <button
            on:click|preventDefault={resetConfigParams}
            class="absolute right-1 bottom-4 bg-transparent hover:bg-red-500 text-red-600 text-xs hover:text-white py-1 px-1 border border-red-500 hover:border-transparent rounded transition dark:text-red-400 dark:border-red-400">
            Reset
          </button>
        {/if}

      </div>


      {#if currentImgHash != null }
        <div class="flex flex-col gap-6">
          {#if currentTab === 'emotions'}
            <div class="flex w-full justify-around flex-wrap gap-2">
              <div class="w-[10rem]">
                <div>Happy</div>
                <input class="range-slider" type="range" min="0" max="1" step="0.01" bind:value={happyValue}>
                <span>{happyValue}</span>
              </div>
              <div class="w-[10rem]">
                <div>Sad</div>
                <input class="range-slider" type="range" min="0" max="1" step="0.01" bind:value={sadValue}>
                <span>{sadValue}</span>
              </div>
              <div class="w-[10rem]">
                <div>Fear</div>
                <input class="range-slider" type="range" min="0" max="1" step="0.01" bind:value={fearValue}>
                <span>{fearValue}</span>
              </div>
              <div class="w-[10rem]">
                <div>Anger</div>
                <input class="range-slider" type="range" min="0" max="1" step="0.01" bind:value={angerValue}>
                <span>{angerValue}</span>
              </div>
              <div class="w-[10rem]">
                <div>Surprise</div>
                <input class="range-slider" type="range" min="0" max="1" step="0.01" bind:value={surpriseValue}>
                <span>{surpriseValue}</span>
              </div>
              <div class="w-[10rem]">
                <div>Disgust</div>
                <input class="range-slider" type="range" min="0" max="1" step="0.01" bind:value={disgustValue}>
                <span>{disgustValue}</span>
              </div>
            </div>
            <button class="simple-button" on:click={applyEmotionFunc}>Apply emotions</button>
          {:else if currentTab === 'ez'}
            <label class="relative inline-flex items-center cursor-pointer my-2">
              <input type="checkbox" bind:checked="{showFaceTexture}" class="sr-only peer">
              <div
                class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-500 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
              <span class="ml-3 text-sm font-medium text-gray-900 dark:text-gray-300 ">Show face texture</span>
            </label>
            <div class="flex gap-32">
              <div>
                <div class="font-bold">Mouth</div>
                <RangeSlider bind:values={mouthOpenness} min={0} max={3} vertical pips reversed all="label"
                             formatter={(value) => mountOpennessInfo[value].label} />
              </div>
              <div>
                <div class="font-bold">Eyebrows</div>
                <RangeSlider bind:values={eyebrowsRaise} min={-1} max={2} vertical pips all="label"
                             formatter={(value) => eyebrowsRaiseInfo[value].label} />
              </div>
            </div>
            <div class="flex flex-col items-center">
              <div class="font-bold">Head pose</div>
              <div class="flex flex-col items-center">
                Up
                <div class="flex gap-1 items-center">
                  Right
                  <planar-range>
                    <planar-range-thumb
                      x={lookDirection[0]}
                      y={lookDirection[1]}
                      on:change={({detail}) => {
                    lookDirection = [detail.x, detail.y];
                  }}
                    />
                  </planar-range>
                  Left
                </div>
                Down
              </div>
            </div>
            <div>
              <div class="font-bold">Hair color</div>
              <div class="flex gap-2">
                <button class="simple-button" on:click={() => {
                  hairColor = '#ffcc99';
                }}>Blonde
                </button>
                <button class="simple-button" on:click={() => {
                  hairColor = '#4c331a';
                }}>Brunette
                </button>
                <button class="simple-button" on:click={() => {
                  hairColor = '#4f1414';
                }}>Red
                </button>
                <button class="simple-button" on:click={() => {
                  hairColor = '#5d5d5d';
                }}>Black
                </button>
              </div>
            </div>
            <button class="simple-button" on:click={applyManualValuesFunc}>Apply expression</button>
          {:else if currentTab === 'manual'}
            <button class="simple-button" on:click={applyManualValuesFunc}>Apply manual FLAME sliders</button>
            <label class="relative inline-flex items-center cursor-pointer my-2">
              <input type="checkbox" bind:checked="{showFaceTexture}" class="sr-only peer">
              <div
                class="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 rounded-full peer dark:bg-gray-500 peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
              <span class="ml-3 text-sm font-medium text-gray-900 dark:text-gray-300 ">Show face texture</span>
            </label>
            <div>
              Hair color
              <input type="color" bind:value={hairColor} />

            </div>
            <div class="flex flex-col gap-8">
              <div class="flex gap-2 flex-wrap justify-around">
                {#each expArr as value, i}
                  <div class="w-[10rem]">
                    <div>Expression {i + 1}</div>
                    <input class="range-slider" type="range" min="-5" max="5" step="0.01" bind:value={value}>
                    <span>{value.toFixed(2)}</span>
                  </div>
                {/each}
              </div>
              <div class="flex gap-2 flex-wrap justify-around">
                {#each poseArr as value, i}
                  <div class="w-[10rem]">
                    <div>{poseArrLabels[i]}</div>
                    <input class="range-slider" type="range" min="-5" max="5" step="0.01" bind:value={value}>
                    <span>{value.toFixed(2)}</span>
                  </div>
                {/each}
              </div>
              <div class="flex gap-2 flex-wrap justify-around">
                {#each neckPoseArr as value, i}
                  <div class="w-[10rem]">
                    <div>{neckPoseArrLabels[i]}</div>
                    <input class="range-slider" type="range" min="-5" max="5" step="0.01" bind:value={value}>
                    <span>{value.toFixed(2)}</span>
                  </div>
                {/each}
              </div>
              <div class="flex gap-2 flex-wrap justify-around">
                {#each eyePoseArr as value, i}
                  <div class="w-[10rem]">
                    <div>Eye {i + 1}</div>
                    <input class="range-slider" type="range" min="-5" max="5" step="0.01" bind:value={value}>
                    <span>{value.toFixed(2)}</span>
                  </div>
                {/each}
              </div>
            </div>
          {/if}
        </div>
      {/if}

    </div>
  </div>
</main>

<style lang="postcss">
    .range-slider {
        @apply w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer;
    }

    .simple-button {
        @apply px-5 py-2.5 font-medium bg-blue-50 hover:bg-blue-100 hover:text-blue-600 text-blue-500 rounded-lg text-sm
        dark:bg-slate-500 dark:hover:bg-slate-600 dark:text-gray-200 dark:hover:text-gray-100;
    }

    :global(.rangePips.vertical .pipVal) {
        transform: translate(0.5rem, -50%) !important;
    }

    :global(.rangePips .pipVal) {
        @apply dark:text-slate-200;
    }

    planar-range {
        border: 0.1rem solid #818181;
        background: url("/grid.svg") no-repeat #e5e5e5;
        background-size: cover;
        width: 10rem;
        height: 10rem;
    }

    planar-range-thumb {
        width: 1.5rem;
        height: 1.5rem;
    }

    ::-webkit-scrollbar-track {
        background-color: #d2d2d2;
    }

    ::-webkit-scrollbar {
        width: 0.5rem;
        background-color: #F5F5F5;
    }

    ::-webkit-scrollbar-thumb {
        background-color: #4b4b4b;
        border: 2px solid #555555;
        border-radius: 200px;
    }
</style>