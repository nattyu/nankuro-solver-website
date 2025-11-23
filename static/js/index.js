// static/js/index.js

document.addEventListener("DOMContentLoaded", () => {
  const imgInput      = document.getElementById("imgInput");
  const runBtn        = document.getElementById("runBtn");
  const overlay       = document.getElementById("progressOverlay");
  const bar           = document.getElementById("progressBar");
  const txt           = document.getElementById("progressText");
  const imgPreview    = document.getElementById("imgPreview"); // あれば使う（なければ無視）

  // 初期状態
  if (overlay) {
    overlay.style.display = "none";
  }
  if (runBtn) {
    runBtn.disabled = true;
  }

  // プログレス更新
  function setProgress(p) {
    const v = Math.max(0, Math.min(100, p));
    if (bar) {
      bar.style.width = v + "%";
    }
    if (txt) {
      txt.textContent = v + "%";
    }
  }

  // 画像選択時のプレビューとボタン有効化
  if (imgInput) {
    imgInput.addEventListener("change", () => {
      const file = imgInput.files[0];

      if (!file) {
        if (imgPreview) {
          imgPreview.classList.add("d-none");
        }
        if (runBtn) {
          runBtn.disabled = true;
        }
        return;
      }

      // プレビュー（imgPreview がある場合のみ）
      if (imgPreview) {
        const reader = new FileReader();
        reader.onload = (e) => {
          imgPreview.src = e.target.result;
          imgPreview.classList.remove("d-none");
        };
        reader.readAsDataURL(file);
      }

      if (runBtn) {
        runBtn.disabled = false;
      }
    });
  }

  // OCR 実行
  if (runBtn) {
    runBtn.addEventListener("click", () => {
      if (!imgInput || !imgInput.files.length) {
        alert("画像を選択してください。");
        return;
      }

      const originalFile = imgInput.files[0];

      // UI ロック & プログレス初期化
      runBtn.disabled = true;
      imgInput.disabled = true;
      if (overlay) {
        overlay.style.display = "flex";
      }
      setProgress(0);

      const formData = new FormData();
      // ★ 画像だけ送る。mask / corners はもう使わない
      formData.append("image", originalFile);

      fetch("/process", { method: "POST", body: formData })
        .then((res) => {
          if (!res.body) {
            throw new Error("ストリーミングレスポンスが利用できません。");
          }

          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let buffer = "";
          let resultHtml = "";

          function readChunk() {
            return reader.read().then(({ done, value }) => {
              if (done) {
                if (overlay) {
                  overlay.style.display = "none";
                }
                // 最後に HTML があればそのまま描画
                if (resultHtml) {
                  window.history.pushState({}, "", "/result");
                  document.open();
                  document.write(resultHtml);
                  document.close();
                }
                // UI 解放
                runBtn.disabled = false;
                imgInput.disabled = false;
                return;
              }

              const chunk = decoder.decode(value, { stream: true });
              buffer += chunk;
              const lines = buffer.split("\n");
              buffer = lines.pop(); // 最後の行は途中かもしれないので次回へ

              for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed) continue;

                // まず JSON として解釈（progress / error）
                let obj = null;
                try {
                  obj = JSON.parse(trimmed);
                } catch (e) {
                  obj = null;
                }

                if (obj) {
                  if (obj.progress !== undefined) {
                    setProgress(obj.progress);
                  }
                  if (obj.error) {
                    // エラーなら即通知して終了
                    if (overlay) {
                      overlay.style.display = "none";
                    }
                    runBtn.disabled = false;
                    imgInput.disabled = false;
                    alert(obj.error);
                    // これ以上読んでも意味がないのでストリームを止める
                    reader.cancel();
                    return;
                  }
                } else {
                  // JSON でない行は result.html の本体とみなす
                  resultHtml += line + "\n";
                }
              }

              return readChunk();
            });
          }

          return readChunk();
        })
        .catch((err) => {
          console.error(err);
          if (overlay) {
            overlay.style.display = "none";
          }
          runBtn.disabled = false;
          if (imgInput) {
            imgInput.disabled = false;
          }
          alert(`処理中にエラーが発生しました:\n${err.message}`);
        });
    });
  }
});
