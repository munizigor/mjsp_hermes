document.addEventListener('DOMContentLoaded', () => {
    const goTopButton = document.getElementById('go-top-button');

    if (!goTopButton) {
        return;
    }

    function toggleButtonVisibility() {
        if (window.scrollY > 50) {
            goTopButton.classList.add('is-visible');
        } else {
            goTopButton.classList.remove('is-visible');
        }
    }

    window.addEventListener('scroll', toggleButtonVisibility, { passive: true });

    goTopButton.addEventListener('click', () => {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });

    toggleButtonVisibility();
});
