if (!window.presentacionLightboxListener) {
  window.presentacionLightboxListener = true;
  let lightbox;

  document.addEventListener("mkdocs-revealjs-ready", (event) => {
    if (!event.detail.root.querySelector(".presentacion-lightbox")) return;
    if (lightbox) lightbox.destroy();
    lightbox = GLightbox({ selector: ".presentacion-lightbox" });
  });
}
