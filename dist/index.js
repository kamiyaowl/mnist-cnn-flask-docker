const container = new Vue({
    el: '#container',
    data: {
        message: '0から9の数字を書いたら識別します！',
        pen_size: "50",
        is_debug: false,
    },
    methods: {
        update_message: function(str) {
            this.message = str;
        },
        clear: function() {
            const canvas = document.getElementById('draw_canvas');
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            this.update_message('また書いてね!');
        },
        touch_draw: function(e) {
            const rect = e.target.getBoundingClientRect();
            // 気分でマルチタッチ対応してみる
            for(const t of e.touches) {
                const x = t.clientX - rect.left;
                const y = t.clientY - rect.top;
                this.draw(x, y);
            }
        },
        drag_draw: function(e) {
            if(!e.buttons) return;

            const rect = e.target.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            this.draw(x, y);
        },
        draw: function(mx, my) {
            const canvas = document.getElementById('draw_canvas');
            // 表示サイズとcanvasサイズは異なるので変換しておく
            const x = mx / canvas.clientWidth * canvas.width;
            const y = my / canvas.clientHeight * canvas.height;
            if (x < 0 || y < 0 || canvas.width < x || canvas.height < y) return;
            // 点を書く
            const ctx = canvas.getContext('2d');
            const r = parseFloat(this.pen_size) / 100.0 * (canvas.width / 8);
            ctx.beginPath();
            ctx.fillStyle = 'white';
            ctx.arc(x, y, r, 0, Math.PI * 2, true);
            ctx.fill();
        },
        predict: function() {
            const canvas = document.getElementById('draw_canvas');
            const ctx = canvas.getContext('2d');
            // RGBA32
            const img = ctx.getImageData(0,0,28,28).data;
            const length = img.length / 4;
            // とりあえず面倒なので加重平均とかはしない
            const src = [];
            for(let i = 0 ; i < length ; ++i) {
                const ptr = i * 4;
                src.push(Math.floor((img[ptr] + img[ptr + 1] + img[ptr + 2]) / 3.0));
            }
            // flaskで作った推論機に投げる
            callback = this.update_message; // then内でthis参照させるのがかったるい
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({'src': src }),
            }).then(function(res) {
                return res.json();
            }).then(function(data) {
                // predict[1][20], elapsed[sec]が帰ってくるので適当に表示する
                const predict = data.predict[0];
                let index = 0;
                for(let i = 0 ; i < predict.length ; ++i) {
                    if (predict[index] < predict[i]) {
                        index = i;
                    }
                }
                callback(`たぶん${index}だと思う。(${Math.floor(predict[index] * 100)}% ${data.elapsed}[sec])`);
            });
            // 確認用
            // this.debug_print(src, null);
        },
        debug_print: function(src, predict) {
            if (this.is_debug) {
                let debug = "";
                for(let j = 0 ; j < 28 ; ++j) {
                    for(let i = 0 ; i < 28 ; ++i) {
                        debug += `  ${src[j * 28 + i].toString(16)} `.slice(-3);
                    }
                    debug += '\r\n';
                }
                console.log(debug);
                console.log(predict);
            }
        }
    },
});