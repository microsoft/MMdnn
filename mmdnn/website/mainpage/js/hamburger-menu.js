function checkForHamburgerMenu() {
    var mobileNavOpenBtn = document.getElementById('mobile-nav-open');
    var mobileNav = document.getElementById('mobile-nav');

    if (mobileNavOpenBtn && mobileNav) {
        initMobileNav();
    } else {
        console.log("can't find it");
        setTimeout(checkForHamburgerMenu, 100);
    }
}

function initMobileNav() {
    var mobileNavOpenBtn = document.getElementById('mobile-nav-open');
    var mobileNav = document.getElementById('mobile-nav');
    var toggleMobileNav = function (e) {
        if (mobileNav.classList.contains('open')) {
            mobileNav.classList.remove('open');
            mobileNavOpenBtn.classList.remove('open');
        } else {
            mobileNav.classList.add('open');
            mobileNavOpenBtn.classList.add('open');
        }
        e.stopPropagation();
    };
    mobileNavOpenBtn.addEventListener('click', toggleMobileNav);
    mobileNav.addEventListener('click', toggleMobileNav);
    if (window.location.search !== '') {
        window.history.replaceState({}, '', window.location.origin);
    }

    var video = document.getElementById("covervid");
    if (video) video.play();
}

checkForHamburgerMenu();
