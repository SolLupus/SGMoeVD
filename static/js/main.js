// 主要JavaScript文件

// 等待DOM加载完成
document.addEventListener('DOMContentLoaded', function() {
    // 初始化代码编辑器
    const codeEditor = CodeMirror.fromTextArea(document.getElementById('codeEditor'), {
        mode: 'text/x-csrc',  // C语言语法高亮
        theme: 'dracula',     // 使用暗色主题
        lineNumbers: true,    // 显示行号
        indentUnit: 4,        // 缩进单位
        smartIndent: true,    // 智能缩进
        tabSize: 4,           // Tab大小
        indentWithTabs: false,// 使用空格缩进
        lineWrapping: true,   // 长行换行
        matchBrackets: true,  // 匹配括号
        autoCloseBrackets: true, // 自动闭合括号
        highlightSelectionMatches: true // 高亮选中文本的匹配项
    });

    // 获取元素
    const codeForm = document.getElementById('codeForm');
    const loadExampleBtn = document.getElementById('loadExample');
    const clearCodeBtn = document.getElementById('clearCode');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultsCard = document.getElementById('resultsCard');
    const graphsCard = document.getElementById('graphsCard');
    const expertsCard = document.getElementById('expertsCard');
    const resultIcon = document.getElementById('resultIcon');
    const resultText = document.getElementById('resultText');
    const confidenceBar = document.getElementById('confidenceBar');
    
    // 图像元素
    const pdgGraph = document.getElementById('pdgGraph');
    const cfgGraph = document.getElementById('cfgGraph');
    const astGraph = document.getElementById('astGraph');

    // 专家模型进度条
    const pdgExperts = [
        document.getElementById('pdgExpert1'),
        document.getElementById('pdgExpert2'),
        document.getElementById('pdgExpert3'),
        document.getElementById('pdgExpert4')
    ];
    
    const cfgExperts = [
        document.getElementById('cfgExpert1'),
        document.getElementById('cfgExpert2'),
        document.getElementById('cfgExpert3'),
        document.getElementById('cfgExpert4')
    ];
    
    const astExperts = [
        document.getElementById('astExpert1'),
        document.getElementById('astExpert2'),
        document.getElementById('astExpert3'),
        document.getElementById('astExpert4')
    ];
    
    // 加载示例代码
    loadExampleBtn.addEventListener('click', function() {
        fetch('/get_example_code')
            .then(response => response.json())
            .then(data => {
                codeEditor.setValue(data.code);
            })
            .catch(error => {
                console.error('加载示例代码失败:', error);
            });
    });
    
    // 清空代码
    clearCodeBtn.addEventListener('click', function() {
        codeEditor.setValue('');
    });
    
    // 提交表单进行分析
    codeForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // 获取代码
        const code = codeEditor.getValue();
        
        if (!code.trim()) {
            alert('请输入或选择代码');
            return;
        }
        
        // 显示加载动画
        analyzeBtn.disabled = true;
        loadingSpinner.classList.remove('d-none');
        analyzeBtn.textContent = '正在分析...';
        
        // 隐藏之前的结果
        resultsCard.style.display = 'none';
        graphsCard.style.display = 'none';
        expertsCard.style.display = 'none';
        
        // 准备表单数据
        const formData = new FormData();
        formData.append('code', code);
        
        // 发送请求
        fetch('/detect', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // 处理响应
            if (data.error) {
                alert('分析过程中出错: ' + data.error);
                return;
            }
            
            // 显示结果卡片
            resultsCard.style.display = 'block';
            graphsCard.style.display = 'block';
            expertsCard.style.display = 'block';
            
            // 更新结果
            updateResults(data);
            
            // 更新图可视化
            updateGraphVisualizations(data.graph_visualizations);
            
            // 更新专家模型使用情况
            updateExpertUsage(data.expert_usage);
        })
        .catch(error => {
            console.error('请求失败:', error);
            alert('请求失败，请检查网络连接');
        })
        .finally(() => {
            // 恢复按钮状态
            analyzeBtn.disabled = false;
            loadingSpinner.classList.add('d-none');
            analyzeBtn.textContent = '分析代码';
        });
    });
    
    // 更新结果显示
    function updateResults(data) {
        // 设置图标和文本
        if (data.prediction === 0) {
            // 安全代码
            resultIcon.innerHTML = '<i class="bi bi-shield-check safe-icon"></i>';
            resultIcon.className = 'display-1 mb-3 safe-icon';
            resultText.textContent = '代码安全 - 未检测到明显漏洞';
            confidenceBar.className = 'progress-bar bg-success';
        } else {
            // 漏洞代码
            resultIcon.innerHTML = '<i class="bi bi-shield-exclamation danger-icon"></i>';
            resultIcon.className = 'display-1 mb-3 danger-icon';
            resultText.textContent = '检测到潜在漏洞';
            confidenceBar.className = 'progress-bar bg-danger';
            
        }
        
        // 更新置信度
        const confidence = data.confidence.toFixed(2);
        confidenceBar.style.width = confidence + '%';
        confidenceBar.setAttribute('aria-valuenow', confidence);
        confidenceBar.textContent = confidence + '%';
    }
    
    // 更新图可视化
    function updateGraphVisualizations(visualizations) {
        // 设置图像源
        pdgGraph.src = 'data:image/png;base64,' + visualizations.pdg;
        cfgGraph.src = 'data:image/png;base64,' + visualizations.cfg;
        astGraph.src = 'data:image/png;base64,' + visualizations.ast;
    }
    
    // 更新专家模型使用情况
    function updateExpertUsage(expertUsage) {
        // 对每个专家值进行归一化
        function normalizeValues(values) {
            const sum = values.reduce((a, b) => a + b, 0);
            return values.map(v => (v / sum) * 100);
        }
        
        const pdgValues = normalizeValues(expertUsage.pdg);
        const cfgValues = normalizeValues(expertUsage.cfg);
        const astValues = normalizeValues(expertUsage.ast);
        const ast_expert_types = ['树结构', '语法', '嵌套',"轻量"] 
        const pdg_expert_types = ['数据流', '变量依赖', '全局依赖',"轻量"] 
        const cfg_expert_types = ['控制流', '路径', '循环',"轻量"] 
        // 更新PDG专家进度条
        pdgValues.forEach((value, i) => {
            pdgExperts[i].style.width = value.toFixed(1) + '%';
            pdgExperts[i].setAttribute('aria-valuenow', value.toFixed(1));
            pdgExperts[i].textContent = pdg_expert_types[i] + '专家' + ' (' + value.toFixed(1) + '%)';
        });
        
        // 更新CFG专家进度条
        cfgValues.forEach((value, i) => {
            cfgExperts[i].style.width = value.toFixed(1) + '%';
            cfgExperts[i].setAttribute('aria-valuenow', value.toFixed(1));
            cfgExperts[i].textContent = cfg_expert_types[i] + '专家'+' (' + value.toFixed(1) + '%)';
        });
        
        // 更新AST专家进度条
        astValues.forEach((value, i) => {
            astExperts[i].style.width = value.toFixed(1) + '%';
            astExperts[i].setAttribute('aria-valuenow', value.toFixed(1));
            astExperts[i].textContent = ast_expert_types[i] + '专家' + ' (' + value.toFixed(1) + '%)';
        });
    }
    
    // 加载模型信息
    function loadModelInfo() {
        fetch('/model_info')
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('modelInfo').innerHTML = `<p class="text-danger">${data.error}</p>`;
                    return;
                }
                
                // 创建模型信息HTML
                const infoHTML = `
                    <div class="table-responsive">
                        <table class="table table-bordered">
                            <tbody>
                                <tr>
                                    <th scope="row">隐藏层大小</th>
                                    <td>${data.hidden_size}</td>
                                    <th scope="row">GCN隐藏层大小</th>
                                    <td>${data.gcn_hidden_size}</td>
                                </tr>
                                <tr>
                                    <th scope="row">GCN层数</th>
                                    <td>${data.gcn_num_layers}</td>
                                    <th scope="row">融合维度</th>
                                    <td>${data.fusion_dim}</td>
                                </tr>
                                <tr>
                                    <th scope="row">专家数量</th>
                                    <td>${data.num_experts}</td>
                                    <th scope="row">使用专业化MoE</th>
                                    <td>${data.specialized_moe ? '是' : '否'}</td>
                                </tr>
                                <tr>
                                    <th scope="row">跨图注意力机制</th>
                                    <td colspan="3">${data.cross_graph_attention ? '启用' : '禁用'}</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    <div class="alert alert-info">
                        <strong>模型特点：</strong> 使用了GCN神经网络对代码的三种图表示进行编码，然后通过混合专家模型进行特征融合和决策。
                    </div>
                `;
                
                document.getElementById('modelInfo').innerHTML = infoHTML;
            })
            .catch(error => {
                console.error('加载模型信息失败:', error);
                document.getElementById('modelInfo').innerHTML = '<p class="text-danger">加载模型信息失败</p>';
            });
    }
    
    // 初始加载模型信息
    loadModelInfo();
    
    // 添加Bootstrap图标库
    const iconLink = document.createElement('link');
    iconLink.rel = 'stylesheet';
    iconLink.href = 'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css';
    document.head.appendChild(iconLink);
}); 