document.addEventListener('DOMContentLoaded', () => {
    const audioPlayer = document.getElementById('audio-player');

    document.querySelectorAll('.play-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const track = this.closest('.track');
            const audioSrc = track.getAttribute('data-src');

            audioPlayer.src = audioSrc;
            audioPlayer.play();

            // Можно добавить выделение текущего трека
            document.querySelectorAll('.track').forEach(t => t.classList.remove('active'));
            track.classList.add('active');
        });
    });
});