// onnxruntime-webへ推論を委譲するためのEmscripten JSライブラリ。
//
// `crates/voicevox_core/src/core/infer/runtimes/onnxruntime_wasm.rs` の
// `unsafe extern "C"` 宣言に対応する実装。`__async: true` と `Asyncify.handleAsync`
// により、Rust側からは同期関数として呼べる。
//
// onnxruntime-web本体はwasmには含めず、実行時に読み込む。既定では
// `Module.ortModule` (呼び出し側が渡したモジュール) を使い、無ければ
// `Module.ortScriptUrl` から動的importする。

addToLibrary({
  $VvOrt: {
    /** onnxruntime-webのモジュール。 */
    ort: null,
    /** セッションID → InferenceSession。 */
    sessions: new Map(),
    nextSessionId: 1,
    lastError: null,

    setError: function (e) {
      VvOrt.lastError = e && e.message ? e.message : String(e);
    },

    /**
     * onnxruntime-webのTensor型名から、1要素あたりのバイト数とTypedArrayを得る。
     * voicevox_coreが扱うのはfloat32とint64のみ。
     */
    typeInfo: function (type) {
      switch (type) {
        case 'float32':
          return { bytes: 4, array: Float32Array };
        case 'int64':
          return { bytes: 8, array: BigInt64Array };
        default:
          throw new Error('unsupported tensor type: ' + type);
      }
    },

    /**
     * `session.inputMetadata` / `outputMetadata` から名前に対応する項目を引く。
     *
     * onnxruntime-web 1.20以降は`inputNames`と同じ並びの配列。将来名前引きの
     * オブジェクトに変わっても動くようにしておく。
     */
    metaFor: function (metadata, name, index) {
      if (Array.isArray(metadata)) {
        return metadata.find((m) => m && m.name === name) ?? metadata[index] ?? null;
      }
      return metadata ? (metadata[name] ?? null) : null;
    },

    /**
     * メタ情報からTensorの型名を取る。
     *
     * 判らないまま既定値に倒すと、Rust側の型検査が「floatが来た」と言って
     * 落ちるだけで原因が判らなくなるため、ここで失敗させる。
     */
    typeOf: function (meta, name) {
      if (meta && typeof meta.type === 'string') return meta.type;
      throw new Error('cannot determine the tensor type of ' + name);
    },

    ndimOf: function (meta) {
      const shape = meta && (meta.shape ?? meta.dims);
      return Array.isArray(shape) ? shape.length : null;
    },
  },

  ort_web_init__deps: ['$VvOrt'],
  ort_web_init__async: true,
  ort_web_init: function () {
    return Asyncify.handleAsync(async function () {
      try {
        if (VvOrt.ort) return 0;

        let ort = Module.ortModule || (typeof globalThis !== 'undefined' && globalThis.ort);
        if (!ort) {
          const url = Module.ortScriptUrl;
          if (!url) {
            throw new Error(
              'onnxruntime-web is not available; ' +
                'set `Module.ortModule` or `Module.ortScriptUrl` before instantiating',
            );
          }
          ort = await import(url);
          if (ort && ort.default && !ort.InferenceSession) ort = ort.default;
        }
        if (Module.ortWasmPaths && ort.env && ort.env.wasm) {
          ort.env.wasm.wasmPaths = Module.ortWasmPaths;
        }
        // cross-origin isolationが無い環境ではSharedArrayBufferが使えないため
        // シングルスレッドに固定する。
        if (ort.env && ort.env.wasm && !globalThis.crossOriginIsolated) {
          ort.env.wasm.numThreads = 1;
        }
        VvOrt.ort = ort;
        return 0;
      } catch (e) {
        VvOrt.setError(e);
        return -1;
      }
    });
  },

  ort_web_session_new__deps: ['$VvOrt', 'malloc'],
  ort_web_session_new__async: true,
  ort_web_session_new: function (modelPtr, modelLen, outMetaJson) {
    return Asyncify.handleAsync(async function () {
      try {
        // wasmヒープからコピーする。onnxruntime-webは自身のヒープへ載せ替えるため、
        // ここでのコピーは推論中は保持されない。
        const bytes = HEAPU8.slice(modelPtr, modelPtr + modelLen);
        const session = await VvOrt.ort.InferenceSession.create(bytes, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'basic',
        });

        const id = VvOrt.nextSessionId++;
        VvOrt.sessions.set(id, session);

        const describe = (names, metadata) =>
          names.map((name, i) => {
            const meta = VvOrt.metaFor(metadata, name, i);
            return {
              name: name,
              type: VvOrt.typeOf(meta, name),
              ndim: VvOrt.ndimOf(meta),
            };
          });

        const meta = {
          inputs: describe(session.inputNames, session.inputMetadata),
          outputs: describe(session.outputNames, session.outputMetadata),
        };
        HEAPU32[outMetaJson >> 2] = stringToNewUTF8(JSON.stringify(meta));
        return id;
      } catch (e) {
        VvOrt.setError(e);
        return -1;
      }
    });
  },

  ort_web_session_release__deps: ['$VvOrt'],
  ort_web_session_release: function (sessionId) {
    const session = VvOrt.sessions.get(sessionId);
    if (!session) return;
    VvOrt.sessions.delete(sessionId);
    // `release`はPromiseを返すが、解放は待たなくてよい。
    if (typeof session.release === 'function') {
      session.release().catch(function () {});
    }
  },

  ort_web_session_run__deps: ['$VvOrt', 'malloc'],
  ort_web_session_run__async: true,
  ort_web_session_run: function (sessionId, inputsJson, outOutputsJson) {
    return Asyncify.handleAsync(async function () {
      try {
        const session = VvOrt.sessions.get(sessionId);
        if (!session) throw new Error('no such session: ' + sessionId);

        const specs = JSON.parse(UTF8ToString(inputsJson));
        const feeds = {};
        for (const spec of specs) {
          const info = VvOrt.typeInfo(spec.type);
          const count = spec.byteLength / info.bytes;
          // wasmヒープ上のバイト列をTypedArrayとして読み直す。
          // `slice`でコピーするのは、推論中にヒープが伸びてビューが無効化されるのを避けるため。
          const data = new info.array(HEAPU8.buffer, spec.ptr, count).slice();
          feeds[spec.name] = new VvOrt.ort.Tensor(spec.type, data, spec.dims);
        }

        const results = await session.run(feeds);

        const outputs = [];
        for (const name of session.outputNames) {
          const tensor = results[name];
          if (!tensor) throw new Error('missing output: ' + name);
          const info = VvOrt.typeInfo(tensor.type);
          const byteLength = tensor.data.length * info.bytes;
          const ptr = _malloc(byteLength);
          if (!ptr) throw new Error('out of memory');
          HEAPU8.set(new Uint8Array(tensor.data.buffer, tensor.data.byteOffset, byteLength), ptr);
          outputs.push({ type: tensor.type, dims: Array.from(tensor.dims), ptr: ptr });
        }

        HEAPU32[outOutputsJson >> 2] = stringToNewUTF8(JSON.stringify(outputs));
        return 0;
      } catch (e) {
        VvOrt.setError(e);
        return -1;
      }
    });
  },

  ort_web_take_last_error__deps: ['$VvOrt', 'malloc'],
  ort_web_take_last_error: function () {
    const message = VvOrt.lastError;
    VvOrt.lastError = null;
    if (message == null) return 0;
    return stringToNewUTF8(message);
  },
});
