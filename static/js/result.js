// static/js/result.js

document.addEventListener('DOMContentLoaded', () => {
  const form    = document.getElementById('solverForm');
  const overlay = document.getElementById('progressOverlay');
  const bar     = document.getElementById('progressBar');
  const txt     = document.getElementById('progressText');

  form.addEventListener('submit', e => {
    e.preventDefault();

    // プログレスバー初期化＆表示
    overlay.style.display    = 'flex';
    bar.style.width          = '0%';
    txt.textContent          = '0%';

    const formData = new FormData(form);

    fetch(form.action, {
      method: 'POST',
      body: formData
    }).then(res => {
      const reader  = res.body.getReader();
      const decoder = new TextDecoder();
      let resultHtml = '';

      function readChunk() {
        return reader.read().then(({ done, value }) => {
          if (done) {
            // 全部読み終わったら結果ページに書き換え
            document.open();
            document.write(resultHtml);
            document.close();
            return;
          }

          const chunk = decoder.decode(value, { stream: true });
          chunk.split('\n').forEach(line => {
            if (!line.trim()) return;
            try {
              const obj = JSON.parse(line);
              if (obj.progress !== undefined) {
                bar.style.width     = obj.progress + '%';
                txt.textContent     = obj.progress + '%';
              }
              if (obj.error) {
                throw new Error(obj.error);
              }
            } catch (err) {
              // JSON でなければ HTML 断片とみなして連結
              resultHtml += line + '\n';
            }
          });

          return readChunk();
        });
      }

      return readChunk();
    }).catch(err => {
      overlay.style.display = 'none';
      console.error(err);
      alert(`処理中にエラーが発生しました:\n${err.message}`);
    });
  });
});
