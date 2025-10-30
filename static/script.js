document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("solve-form");
  const hiddenInput = document.getElementById("table_json");

  form.addEventListener("submit", function (e) {
    const table = document.getElementById("df_table");
    const rows = table.querySelectorAll("tr");

    const data = Array.from(rows).map(tr =>
      Array.from(tr.querySelectorAll("td")).map(td =>
        td.innerText.trim()
      )
    );

    hiddenInput.value = JSON.stringify(data);
  });
});
