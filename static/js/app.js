// 绘制框提示（box prompt）标注框
// 修复 Issue #3：在（部分）高亮背景上叠加绘制时标注框发生旋转。
// 框提示是轴对齐矩形，绘制时不能引入基于宽高比的旋转角度，
// 否则会破坏画布变换，使标注框看起来被旋转。
function drawBoxPrompt(ctx, box) {
    const x1 = box[0];
    const y1 = box[1];
    const x2 = box[2];
    const y2 = box[3];
    const w = x2 - x1;
    const h = y2 - y1;
    ctx.save();
    // 保持轴对齐绘制，不做 translate/rotate 变换
    ctx.strokeRect(x1, y1, w, h);
    ctx.restore();
}
