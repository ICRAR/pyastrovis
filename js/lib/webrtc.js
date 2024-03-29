module.exports = {
    create_rtc: function(video) {
        let config = {
            sdpSemantics: 'unified-plan'
        };

        config.iceServers = [{urls: ['stun:stun.l.google.com:19302?transport=tcp']}];

        let pc = new RTCPeerConnection();

        // register some listeners to help debugging
        pc.addEventListener('icegatheringstatechange', function () {
            console.warn(pc.iceGatheringState);
        }, false);

        pc.addEventListener('iceconnectionstatechange', function () {
            console.warn(pc.iceConnectionState);
        }, false);

        pc.addEventListener('signalingstatechange', function () {
            console.warn(pc.signalingState);
        }, false);

        // connect audio / video
        pc.addEventListener('track', function (evt) {
            if (evt.track.kind === 'video') {
                video.srcObject = evt.streams[0];
            }
        });

        return pc;
    },

    negotiate_rtc: function(pc, id, url) {
        return pc.createOffer({offerToReceiveVideo: true, offerToReceiveAudio: false}).then(function (offer) {
            return pc.setLocalDescription(offer);
        }).then(function () {
            // wait for ICE gathering to complete
            return new Promise(function (resolve) {
                if (pc.iceGatheringState === 'complete') {
                    resolve();
                } else {
                    function checkState() {
                        if (pc.iceGatheringState === 'complete') {
                            pc.removeEventListener('icegatheringstatechange', checkState);
                            resolve();
                        }
                    }

                    pc.addEventListener('icegatheringstatechange', checkState);
                }
            });
        }).then(function () {
            let offer = pc.localDescription;
            return fetch(url + '/offer', {
                body: JSON.stringify({
                    sdp: offer.sdp,
                    type: offer.type,
                }),
                headers: {
                    'ID': id,
                    'Content-Type': 'application/json'
                },
                method: 'POST'
            });
        }).then(function (response) {
            return response.json();
        }).then(function (answer) {
            return pc.setRemoteDescription(answer);
        });
    }
};

/*function start() {
    document.getElementById('start').style.display = 'none';
    negotiate();
    document.getElementById('stop').style.display = 'inline-block';
}

function stop() {
    document.getElementById('stop').style.display = 'none';
    // close peer connection
    setTimeout(function() {
        pc.close();
    }, 500);
}*/
