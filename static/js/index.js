const cursorGlow = document.querySelector(".cursor-glow");
const hamburger = document.getElementById("hamburger");
const navLinks = document.getElementById("nav-links");
const backToTop = document.getElementById("back-to-top");
const copyBibtexBtn = document.getElementById("copy-bibtex");
const bibtexCode = document.getElementById("bibtex-code");
const themeToggle = document.getElementById("theme-toggle");
const themeIcon = themeToggle ? themeToggle.querySelector("i") : null;
const lightbox = document.getElementById("lightbox");
const lightboxImg = document.getElementById("lightbox-img");
const lightboxTitle = document.getElementById("lightbox-title");
const lightboxClose = document.querySelector(".lightbox-close");
const lightboxPrev = document.querySelector(".lightbox-prev");
const lightboxNext = document.querySelector(".lightbox-next");
const lightboxTriggers = Array.from(document.querySelectorAll(".lightbox-trigger"));

let currentLightboxIndex = 0;

const storedTheme = localStorage.getItem("theme");
if (storedTheme === "dark") {
  document.documentElement.setAttribute("data-theme", "dark");
  if (themeIcon) {
    themeIcon.className = "fas fa-sun";
  }
}

if (cursorGlow) {
  document.addEventListener("mousemove", (event) => {
    cursorGlow.style.left = `${event.clientX}px`;
    cursorGlow.style.top = `${event.clientY}px`;
  });
}

if (hamburger && navLinks) {
  hamburger.addEventListener("click", () => {
    hamburger.classList.toggle("active");
    navLinks.classList.toggle("active");
  });

  navLinks.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      hamburger.classList.remove("active");
      navLinks.classList.remove("active");
    });
  });
}

document.querySelectorAll(".dropdown").forEach((dropdown) => {
  const dropbtn = dropdown.querySelector(".dropbtn");
  if (!dropbtn) {
    return;
  }
  dropbtn.addEventListener("click", (event) => {
    if (window.innerWidth <= 768) {
      event.preventDefault();
      dropdown.classList.toggle("active");
    }
  });
});

if (themeToggle) {
  themeToggle.addEventListener("click", () => {
    const isDark = document.documentElement.getAttribute("data-theme") === "dark";
    if (isDark) {
      document.documentElement.removeAttribute("data-theme");
      localStorage.removeItem("theme");
      if (themeIcon) {
        themeIcon.className = "fas fa-moon";
      }
    } else {
      document.documentElement.setAttribute("data-theme", "dark");
      localStorage.setItem("theme", "dark");
      if (themeIcon) {
        themeIcon.className = "fas fa-sun";
      }
    }
  });
}

window.addEventListener("scroll", () => {
  if (!backToTop) {
    return;
  }
  if (window.scrollY > 300) {
    backToTop.classList.add("show");
  } else {
    backToTop.classList.remove("show");
  }
});

if (backToTop) {
  backToTop.addEventListener("click", () => {
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
}

if (copyBibtexBtn && bibtexCode) {
  copyBibtexBtn.addEventListener("click", async () => {
    const text = bibtexCode.innerText.trim();
    await navigator.clipboard.writeText(text);
    copyBibtexBtn.textContent = "Copied";
    setTimeout(() => {
      copyBibtexBtn.textContent = "Copy";
    }, 1500);
  });
}

function openLightbox(index) {
  const trigger = lightboxTriggers[index];
  if (!trigger) {
    return;
  }
  currentLightboxIndex = index;
  lightboxImg.src = trigger.src;
  lightboxTitle.textContent = trigger.dataset.title || trigger.alt || "";
  lightbox.classList.add("active");
  document.body.style.overflow = "hidden";
}

function closeLightbox() {
  lightbox.classList.remove("active");
  document.body.style.overflow = "";
}

function navigateLightbox(direction) {
  if (!lightboxTriggers.length) {
    return;
  }
  currentLightboxIndex = (currentLightboxIndex + direction + lightboxTriggers.length) % lightboxTriggers.length;
  openLightbox(currentLightboxIndex);
}

lightboxTriggers.forEach((trigger, index) => {
  trigger.addEventListener("click", () => openLightbox(index));
});

if (lightboxClose) {
  lightboxClose.addEventListener("click", closeLightbox);
}

if (lightboxPrev) {
  lightboxPrev.addEventListener("click", () => navigateLightbox(-1));
}

if (lightboxNext) {
  lightboxNext.addEventListener("click", () => navigateLightbox(1));
}

if (lightbox) {
  lightbox.addEventListener("click", (event) => {
    if (event.target === lightbox) {
      closeLightbox();
    }
  });
}

document.addEventListener("keydown", (event) => {
  if (!lightbox || !lightbox.classList.contains("active")) {
    return;
  }
  if (event.key === "Escape") {
    closeLightbox();
  }
  if (event.key === "ArrowLeft") {
    navigateLightbox(-1);
  }
  if (event.key === "ArrowRight") {
    navigateLightbox(1);
  }
});
