js = """
function createGradioAnimation() {
     
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
/* Top banner styling */
#top-banner {
    padding: 14px 20px;
    border-radius: 12px;
    margin: 12px auto 16px auto;
    width: 100%;
    max-width: 100%;
    font-size: 15px;
    font-weight: 500;
    text-align: center;
    line-height: 1.5;
    transition: all 0.3s ease;
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    box-sizing: border-box;
}

/* Dark mode */
@media (prefers-color-scheme: dark) {
    #top-banner {
        background-color: rgba(42, 47, 74, 0.85);
        color: #e0e0e0;
    }
}

/* Light mode */
@media (prefers-color-scheme: light) {
    #top-banner {
        background-color: rgba(240, 240, 240, 0.9);
        color: #222;
    }
}

.checkbox_info {
    color: var(--block-title-text-color) !important; 
    font-size: var(--block-title-text-size) !important; 
    font-weight: var(--block-title-text-weight) !important;
    height: 22px;
    margin-bottom: 8px !important;
}

::selection {
    background: #ffc078 !important;
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
    height: 50px !important;
    background-color: transparent !important;
    display: flex;
    justify-content: center;
    align-items: center;
}

footer * {
    display: none !important;
}
"""

top_html = """
<div id="top-banner">
    This software is open source under the MIT license. The author does not have any control over the software. Users who use the software and distribute the sounds exported by the software are solely responsible. See the root directory <strong>Agreement-LICENSE</strong> for details.
</div>
<div align="center" style="margin-top: 8px;">
    <div style="display: flex; gap: 80px; justify-content: center; flex-wrap: wrap;">
        <a href="https://github.com/RVC-Boss/GPT-SoVITS" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-GPT--SoVITS-blue.svg?style=for-the-badge&logo=github" style="height: 30px;">
        </a>
        <a href="https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e" target="_blank">
            <img src="https://img.shields.io/badge/简体中文-阅读文档-blue?style=for-the-badge&logo=googledocs&logoColor=white" style="height: 30px;">
        </a>
        <a href="https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e" target="_blank">
            <img src="https://img.shields.io/badge/English-READ%20DOCS-blue?style=for-the-badge&logo=googledocs&logoColor=white" style="height: 30px;">
        </a>
        <a href="https://github.com/RVC-Boss/GPT-SoVITS/blob/main/LICENSE" target="_blank">
            <img src="https://img.shields.io/badge/LICENSE-MIT-green.svg?style=for-the-badge&logo=opensourceinitiative" style="height: 30px;">
        </a>
    </div>
</div>
"""
