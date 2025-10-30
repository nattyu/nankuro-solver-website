document.addEventListener('DOMContentLoaded', () => {
  const canvas     = new fabric.Canvas('c');
  const imgInput   = document.getElementById('imgInput');
  const brushSize  = document.getElementById('brushSize');
  const brushValue = document.getElementById('brushValue');
  const runBtn     = document.getElementById('runBtn');
  const overlay    = document.getElementById('progressOverlay');
  const bar        = document.getElementById('progressBar');
  const txt        = document.getElementById('progressText');

  let scale = 1;
  let corners = [];

  // ブラシカーソル設定
  function setBrushCursor(size) {
    const r = size / 2;
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}"><circle cx="${r}" cy="${r}" r="${r}" fill="rgba(255,0,0,0.3)" /></svg>`;
    const url = `url("data:image/svg+xml,${encodeURIComponent(svg)}") ${r} ${r}, auto`;
    canvas.upperCanvasEl.style.cursor = url;
    canvas.lowerCanvasEl.style.cursor = url;
  }

  // ブラシサイズ変更
  brushSize.addEventListener('input', e => {
    const size = parseInt(e.target.value, 10);
    brushValue.textContent = size;
    if (canvas.freeDrawingBrush) {
      canvas.freeDrawingBrush.width = size;
      setBrushCursor(size);
    }
  });

  // 画像読み込み
  imgInput.addEventListener('change', e => {
    corners = [];
    canvas.clear();
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = evt => fabric.Image.fromURL(evt.target.result, img => {
      const maxW = window.innerWidth * 0.9;
      const maxH = window.innerHeight * 1.0;
      const scaleW = maxW / img.width;
      const scaleH = maxH / img.height;
      scale = Math.min(1, scaleW, scaleH);

      img.set({ scaleX: scale, scaleY: scale });
      canvas.setWidth(img.width * scale);
      canvas.setHeight(img.height * scale);
      canvas.setBackgroundImage(img, canvas.renderAll.bind(canvas));
      canvas.isDrawingMode = false;
      canvas.upperCanvasEl.style.cursor = 'default';
      canvas.lowerCanvasEl.style.cursor = 'default';
      alert('まず4点をクリックして透視補正範囲を指定してください');
    });
    reader.readAsDataURL(file);
  });

  // 4点指定
  canvas.on('mouse:down', opt => {
    if (corners.length < 4) {
      const pt = canvas.getPointer(opt.e);
      corners.push([pt.x / scale, pt.y / scale]);
      canvas.add(new fabric.Circle({
        left: pt.x - 5,
        top:  pt.y - 5,
        radius: 5,
        fill:   'lime',
        selectable: false,
        evented: false
      }));
      if (corners.length === 4) {
        alert('4点指定完了。次にマスクを描画してください');
        canvas.isDrawingMode = true;
        canvas.freeDrawingBrush.color = 'rgba(255,0,0,0.3)';
        const size = parseInt(brushSize.value, 10);
        canvas.freeDrawingBrush.width = size;
        setBrushCursor(size);
      }
    }
  });

  // OCR 実行
  runBtn.addEventListener('click', () => {
    if (corners.length !== 4) {
      return alert('透視補正用の4点をすべて指定してください');
    }
    if (!imgInput.files.length) {
      return alert('画像を選択してください');
    }
    const originalFile = imgInput.files[0];

    // マスク抽出
    const bg = canvas.backgroundImage;
    canvas.backgroundImage = null;
    canvas.renderAll();
    canvas.lowerCanvasEl.toBlob(async maskBlob => {
      canvas.backgroundImage = bg;
      canvas.renderAll();

      const formData = new FormData();
      formData.append('image', originalFile);
      formData.append('mask', maskBlob, 'mask.png');
      formData.append('corners', JSON.stringify(corners));

      overlay.style.display = 'flex';
      bar.style.width = '0%';
      txt.textContent = '0%';

      fetch('/process', { method: 'POST', body: formData })
        .then(res => {
          const reader = res.body.getReader();
          const decoder = new TextDecoder();
          let resultHtml = '';

          function readChunk() {
            return reader.read().then(({ done, value }) => {
              if (done) {
                if (resultHtml) {
                  window.history.pushState({}, '', '/result');
                  document.open();
                  document.write(resultHtml);
                  document.close();
                }
                return;
              }
              const chunk = decoder.decode(value, { stream: true });
              chunk.split('\n').forEach(line => {
                if (!line.trim()) return;
                try {
                  const obj = JSON.parse(line);
                  if (obj.progress !== undefined) {
                    bar.style.width = obj.progress + '%';
                    txt.textContent = obj.progress + '%';
                  } else if (obj.error) {
                    throw new Error(obj.error);
                  }
                } catch {
                  resultHtml += line + '\n';
                }
              });
              return readChunk();
            });
          }

          return readChunk();
        })
        .catch(err => {
          overlay.style.display = 'none';
          console.error(err);
          alert(`処理中にエラーが発生しました:\n${err.message}`);
        });
    }, 'image/png', 0.8);
  });
});
