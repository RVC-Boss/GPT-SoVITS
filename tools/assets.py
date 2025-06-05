js = """
function createGradioAnimation() {
    
    const params = new URLSearchParams(window.location.search);
    if (params.get('__theme') !== 'light') { 
        params.set('__theme', 'light'); // 仅当 __theme 不是 'light' 时设置为 'light'
        window.location.search = params.toString(); // 更新 URL，触发页面刷新
    }
    
    var container = document.createElement('div');
    container.id = 'gradio-animation';
    container.style.fontSize = '2em';
    container.style.fontWeight = '500';
    container.style.textAlign = 'center';
    container.style.marginBottom = '20px';
    container.style.fontFamily = '-apple-system, sans-serif, Arial, Calibri';

    var text = 'Welcome to GPT-SoVITS !';
    for (var i = 0; i < text.length; i++) {
        (function(i){
            setTimeout(function(){
                var letter = document.createElement('span');
                letter.style.opacity = '0';
                letter.style.transition = 'opacity 0.5s';
                letter.innerText = text[i];

                container.appendChild(letter);

                setTimeout(function() {
                    letter.style.opacity = '1';
                }, 50);
            }, i * 250);
        })(i);
    }
    return 'Animation created';
}
"""


css = """
/* CSSStyleRule */

.markdown {
    background-color: lightblue;
    padding: 6px 10px;
}

.checkbox_info {
    color: var(--block-title-text-color) !important;
    font-size: var(--block-title-text-size) !important;
    font-weight: var(--block-title-text-weight) !important;
    height: 22px;
    margin-bottom: 8px !important;
}

::selection {
    background: #ffc078; !important;
}

#checkbox_train_dpo input[type="checkbox"]{
    margin-top: 6px;
}

#checkbox_train_dpo span {
    margin-top: 6px;
}

#checkbox_align_train {
    padding-top: 18px;
    padding-bottom: 18px;
}

#checkbox_align_infer input[type="checkbox"] {
    margin-top: 10px;
}

#checkbox_align_infer span {
    margin-top: 10px;
}

footer {
    height: 50px !important;           /* 设置页脚高度 */
    background-color: transparent !important; /* 背景透明 */
    display: flex;
    justify-content: center;           /* 居中对齐 */
    align-items: center;               /* 垂直居中 */
}

footer * {
    display: none !important;          /* 隐藏所有子元素 */
}

"""
top_html = """
<div align="center">
    <div style="margin-bottom: 5px; font-size: 15px;">{}</div>
    <div style="display: flex; gap: 80px; justify-content: center;">
        <a href="https://github.com/RVC-Boss/GPT-SoVITS" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-GPT--SoVITS-blue.svg?style=for-the-badge&logo=github" style="width: auto; height: 30px;">
        </a>
        <a href="https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e" target="_blank">
            <img src="https://img.shields.io/badge/简体中文-阅读文档-blue?style=for-the-badge&logo=googledocs&logoColor=white" style="width: auto; height: 30px;">
        </a>
        <a href="https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e" target="_blank">
            <img src="https://img.shields.io/badge/English-READ%20DOCS-blue?style=for-the-badge&logo=googledocs&logoColor=white" style="width: auto; height: 30px;">
        </a>
        <a href="https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE" target="_blank">
            <img src="https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge&logo=opensourceinitiative" style="width: auto; height: 30px;">
        </a>
    </div>
</div>
"""
