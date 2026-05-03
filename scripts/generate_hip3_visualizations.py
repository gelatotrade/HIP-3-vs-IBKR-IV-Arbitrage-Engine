#!/usr/bin/env python3
"""
Generate animated GIF visualizations for HIP-3 vs IBKR IV Arbitrage.
4 GIFs + 3 PNGs in dark terminal aesthetic matching the original repo.
"""
import matplotlib
matplotlib.use('Agg')
import sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa
import io
from PIL import Image
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hyperliquid_hip3_client import generate_synthetic_hip3_data, load_real_hip3_data
from hip3_rolling_backtest import rolling_backtest_asset, generate_funding_history
from iv_arbitrage_engine import IVArbitrageEngine
from ibkr_options_client import bs_greeks

BG='#080810'; BG2='#0e0e1a'; TEXT='#cccccc'; GREEN='#00ff88'; RED='#ff3344'
YELLOW='#ffaa00'; BLUE='#4488ff'; CYAN='#00ffcc'; WHITE='#ffffff'
GRID_C='#1a1a2e'; DIM='#556677'
REGIME_COLORS={'BULL':GREEN,'NORMAL':BLUE,'CAUTIOUS':YELLOW,'CRISIS':RED,'RECOVERY':CYAN}
CMAP_B=LinearSegmentedColormap.from_list('b',['#0000aa','#0055cc','#22aa66','#00ff88','#eeffaa'])
CMAP_N=LinearSegmentedColormap.from_list('n',['#001144','#003388','#2266aa','#44aacc','#ccffff'])
CMAP_CA=LinearSegmentedColormap.from_list('c',['#440066','#8833aa','#cc7700','#ffaa00','#ffffcc'])
CMAP_CR=LinearSegmentedColormap.from_list('cr',['#000044','#220066','#660088','#cc0022','#ffcc44'])
CMAP_R=LinearSegmentedColormap.from_list('r',['#330044','#553388','#0066aa','#00ccee','#eeffee'])
OUT_DIR=Path(__file__).resolve().parent.parent/'docs'/'img'
DPI=100; TOTAL=60

def _img(fig,w=1400,h=800):
    buf=io.BytesIO()
    fig.savefig(buf,format='png',dpi=DPI,facecolor=fig.get_facecolor(),edgecolor='none',bbox_inches='tight',pad_inches=0.1)
    buf.seek(0); return Image.open(buf).convert('RGBA').resize((w,h),Image.LANCZOS)

def _gif(frames,path,dur=180):
    rgb=[]
    for f in frames:
        bg=Image.new('RGB',f.size,(8,8,16)); bg.paste(f,mask=f.split()[3]); rgb.append(bg)
    rgb[0].save(path,save_all=True,append_images=rgb[1:],duration=dur,loop=0,optimize=True)
    print(f'  Saved {path.name} ({len(rgb)} fr, {path.stat().st_size/1024:.0f}KB)')

def _s(ax):
    ax.set_facecolor(BG2); ax.tick_params(colors=DIM,labelsize=6); ax.grid(True,color=GRID_C,alpha=0.3)
    for s in ax.spines.values(): s.set_color(GRID_C)

def gen_dashboard(data, all_bt):
    print('\n[1/6] hip3_trading_dashboard.gif ...')
    tk='SPY' if 'SPY' in data else list(data.keys())[0]
    df=data[tk]; bt=all_bt[tk]; c=df['close'].values; N=len(c)
    pnl=bt['daily_pnl']; bench=bt['daily_bench']; reg=bt['regimes']
    pos=bt['positions']; afc=bt['arima_fc']; cv=bt['cond_vol']
    cs=np.exp(np.cumsum(pnl))-1; cb=np.exp(np.cumsum(bench))-1
    eq=np.exp(np.cumsum(pnl)); dd=(np.maximum.accumulate(eq)-eq)/np.maximum.accumulate(eq)
    bx=np.linspace(max(40,N//10),N-1,TOTAL,dtype=int); frames=[]
    for fi,d in enumerate(bx):
        fig=plt.figure(figsize=(16,9),facecolor=BG)
        gs=GridSpec(3,2,hspace=0.4,wspace=0.3,left=.06,right=.97,top=.92,bottom=.06)
        rc=REGIME_COLORS.get(reg[d],BLUE); ap=(cs[d]-cb[d])*100
        fig.suptitle(f'LIVE | {tk} | {reg[d]} | Pos:{pos[d]*100:.0f}% | Alpha:{ap:+.1f}%',
                     color=rc,fontsize=13,fontweight='bold',y=.97,fontfamily='monospace')
        dy=np.arange(d+1)
        ax=fig.add_subplot(gs[0,0]); _s(ax)
        ax.plot(dy,cb[:d+1]*100,color='#555588',lw=1.2,label='Buy&Hold')
        ax.plot(dy,cs[:d+1]*100,color=WHITE,lw=2,label='Strategy')
        ax.fill_between(dy,cb[:d+1]*100,cs[:d+1]*100,where=cs[:d+1]>cb[:d+1],color=GREEN,alpha=.1)
        ax.fill_between(dy,cb[:d+1]*100,cs[:d+1]*100,where=cs[:d+1]<=cb[:d+1],color=RED,alpha=.1)
        ax.set_title('Cumulative Return (%)',color=TEXT,fontsize=9,fontweight='bold')
        ax.legend(fontsize=7,loc='upper left',facecolor=BG2,edgecolor=GRID_C,labelcolor=TEXT)
        ax=fig.add_subplot(gs[0,1]); _s(ax)
        for i in range(1,d+1): ax.plot([i-1,i],[c[i-1],c[i]],color=REGIME_COLORS.get(reg[i],BLUE),lw=1.2,alpha=.8)
        ax.set_title(f'{tk} Price',color=TEXT,fontsize=9,fontweight='bold')
        ax=fig.add_subplot(gs[1,0]); _s(ax); fc=afc[:d+1]
        ax.fill_between(dy,0,fc,where=fc>0,color=GREEN,alpha=.3)
        ax.fill_between(dy,0,fc,where=fc<=0,color=RED,alpha=.3)
        ax.plot(dy,fc,color=CYAN,lw=1); ax.axhline(0,color=DIM,lw=.5,ls='--')
        ax.set_title('ARIMA(2,1,2) Forecast',color=TEXT,fontsize=9,fontweight='bold')
        ax=fig.add_subplot(gs[1,1]); _s(ax)
        ax.plot(dy,cv[:d+1]*np.sqrt(365)*100,color=YELLOW,lw=1.2)
        ax.axhline(80,color=RED,lw=.7,ls='--',alpha=.5); ax.axhline(50,color=YELLOW,lw=.7,ls='--',alpha=.5)
        ax.set_title('EWMA Vol (ann%)',color=TEXT,fontsize=9,fontweight='bold')
        ax=fig.add_subplot(gs[2,0]); _s(ax)
        ax.fill_between(dy,pos[:d+1]*100,color=CYAN,alpha=.3); ax.plot(dy,pos[:d+1]*100,color=CYAN,lw=1)
        ax.axhline(100,color=DIM,lw=.5,ls='--'); ax.set_ylim(0,140)
        ax.set_title('Position %',color=TEXT,fontsize=9,fontweight='bold')
        ax=fig.add_subplot(gs[2,1]); _s(ax)
        ax.fill_between(dy,-dd[:d+1]*100,color=RED,alpha=.3); ax.plot(dy,-dd[:d+1]*100,color=RED,lw=1)
        ax.set_title('Drawdown %',color=TEXT,fontsize=9,fontweight='bold')
        frames.append(_img(fig)); plt.close(fig)
        if (fi+1)%15==0: print(f'    {fi+1}/{TOTAL}')
    _gif(frames,OUT_DIR/'hip3_trading_dashboard.gif')

def gen_iv(self=None):
    print('\n[2/6] hip3_iv_surface_3d.gif ...')
    n=40; st=np.linspace(.7,1.3,n); ex=np.linspace(7,180,n); X,Y=np.meshgrid(st,ex); frames=[]
    for fi in range(TOTAL):
        t=fi/TOTAL; biv=.50+.40*np.sin(t*2*np.pi)
        Zi=biv+(-.3)*(X-1)+.4*(X-1)**2+.001*Y
        Zh=biv*1.15+(-.05)*(X-1)+.1*(X-1)**2+.0003*Y
        Zs=Zh-Zi
        fig=plt.figure(figsize=(16,5.5),facecolor=BG)
        gs=GridSpec(1,3,wspace=.08,left=.03,right=.97,top=.88,bottom=.05)
        for idx,(Z,tt,cm) in enumerate([(Zi,'IBKR Options IV',CMAP_N),(Zh,'HIP-3 Implied Vol',CMAP_B),(Zs,'Vol Spread',CMAP_CR)]):
            ax=fig.add_subplot(gs[idx],projection='3d'); ax.set_facecolor(BG)
            ax.xaxis.pane.fill=ax.yaxis.pane.fill=ax.zaxis.pane.fill=False
            for p in[ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]:p.set_edgecolor('#1a1a2e')
            zn=(Z-Z.min())/(Z.max()-Z.min()+1e-10)
            ax.plot_surface(X,Y,Z,facecolors=cm(zn),alpha=.9,rstride=2,cstride=2,edgecolor='none',shade=True)
            ax.set_xlabel('Moneyness',color=DIM,fontsize=6); ax.set_ylabel('DTE',color=DIM,fontsize=6)
            ax.set_title(tt,color=TEXT,fontsize=9,fontweight='bold',pad=2)
            ax.tick_params(colors=DIM,labelsize=5); ax.view_init(elev=25,azim=220+fi*2)
        fig.suptitle(f'IV Surface | Base IV: {biv*100:.0f}%',color=CYAN,fontsize=12,fontweight='bold',y=.96,fontfamily='monospace')
        frames.append(_img(fig,1600,600)); plt.close(fig)
        if (fi+1)%15==0: print(f'    {fi+1}/{TOTAL}')
    _gif(frames,OUT_DIR/'hip3_iv_surface_3d.gif')

def gen_greeks():
    print('\n[3/6] hip3_greeks_surface_3d.gif ...')
    n=35; sr=np.linspace(.8,1.2,n); vr=np.linspace(.2,1.5,n); X,Y=np.meshgrid(sr,vr)
    Zv=np.zeros((n,n)); Zm=np.zeros((n,n)); Zz=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            q=bs_greeks(100*sr[i],100,30/365,.05,vr[j],'call')
            Zv[i,j]=q.vanna; Zm[i,j]=q.vomma; Zz[i,j]=q.zomma
    frames=[]
    for fi in range(TOTAL):
        fig=plt.figure(figsize=(16,5.5),facecolor=BG)
        gs=GridSpec(1,3,wspace=.08,left=.03,right=.97,top=.88,bottom=.05)
        for idx,(Z,tt,cm) in enumerate([(Zv,'Vanna',CMAP_B),(Zm,'Vomma',CMAP_CA),(Zz,'Zomma',CMAP_R)]):
            ax=fig.add_subplot(gs[idx],projection='3d'); ax.set_facecolor(BG)
            ax.xaxis.pane.fill=ax.yaxis.pane.fill=ax.zaxis.pane.fill=False
            for p in[ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane]:p.set_edgecolor('#1a1a2e')
            zn=(Z-Z.min())/(Z.max()-Z.min()+1e-10)
            ax.plot_surface(X,Y,Z,facecolors=cm(zn),alpha=.9,rstride=2,cstride=2,edgecolor='none',shade=True)
            ax.set_xlabel('S/K',color=DIM,fontsize=6); ax.set_ylabel('IV',color=DIM,fontsize=6)
            ax.set_title(tt,color=TEXT,fontsize=9,fontweight='bold',pad=2)
            ax.tick_params(colors=DIM,labelsize=5); ax.view_init(elev=28,azim=210+fi*2.5)
        fig.suptitle('Higher-Order Greeks: Zero on HIP-3 Perps, Non-Zero on IBKR Options',
                     color=GREEN,fontsize=11,fontweight='bold',y=.96,fontfamily='monospace')
        frames.append(_img(fig,1600,600)); plt.close(fig)
        if (fi+1)%15==0: print(f'    {fi+1}/{TOTAL}')
    _gif(frames,OUT_DIR/'hip3_greeks_surface_3d.gif')

def gen_equity(curves):
    print('\n[4/6] hip3_equity_curves.gif ...')
    top=sorted(curves.keys(),key=lambda k:-np.sum(curves[k][0]))[:6]
    cols=['#00ff88','#00ccff','#ff6644','#ffaa00','#cc66ff','#66ffcc']
    ml=max(len(curves[k][0]) for k in top); bx=np.linspace(30,ml-1,TOTAL,dtype=int); frames=[]
    for fi,d in enumerate(bx):
        fig=plt.figure(figsize=(18,12),facecolor=BG)
        gs=GridSpec(3,2,hspace=.35,wspace=.25,left=.06,right=.97,top=.93,bottom=.05)
        fig.suptitle(f'Rolling Backtest Equity Curves | Day {d}',fontsize=14,fontweight='bold',color=WHITE,y=.97)
        for i,tk in enumerate(top):
            s,b=curves[tk]; ax=fig.add_subplot(gs[i//2,i%2]); _s(ax)
            e=min(d+1,len(s)); cs=(np.exp(np.cumsum(s[:e]))-1)*100; cb=(np.exp(np.cumsum(b[:e]))-1)*100
            dy=np.arange(e)
            ax.plot(dy,cb,color='#555588',lw=1.2,label='Bench')
            ax.plot(dy,cs,color=cols[i],lw=1.8,label='Strat')
            ax.fill_between(dy,cb,cs,where=cs>cb,alpha=.12,color=cols[i])
            a=cs[-1]-cb[-1] if len(cs)>0 else 0
            ax.set_title(f'{tk} | Alpha:{a:+.1f}%',fontsize=10,color=WHITE,fontweight='bold')
            ax.legend(fontsize=7,loc='upper left',facecolor=BG2,edgecolor=GRID_C,labelcolor=TEXT)
        frames.append(_img(fig,1600,1000)); plt.close(fig)
        if (fi+1)%15==0: print(f'    {fi+1}/{TOTAL}')
    _gif(frames,OUT_DIR/'hip3_equity_curves.gif')

def gen_regime(data, all_bt):
    print('\n[5/7] hip3_regime_dashboard.gif ...')
    tk='SPY' if 'SPY' in data else list(data.keys())[0]
    df=data[tk]; bt=all_bt[tk]; c=df['close'].values; N=len(c)
    reg=bt['regimes']; pos=bt['positions']
    rets=np.diff(np.log(c),prepend=np.log(c[0]))
    rv=pd.Series(rets).rolling(20,min_periods=5).std().values*np.sqrt(365)*100
    sp_map={'BULL':0.8,'NORMAL':1.0,'CAUTIOUS':2.0,'CRISIS':3.0,'RECOVERY':1.3}
    sp_mult=np.array([sp_map.get(r,1.0) for r in reg])
    N_FRAMES=50
    bx=np.linspace(max(30,N//8),N-1,N_FRAMES,dtype=int); frames=[]
    for fi,d in enumerate(bx):
        fig=plt.figure(figsize=(16,11),facecolor=BG)
        gs=GridSpec(3,2,hspace=0.4,wspace=0.3,left=.06,right=.97,top=.93,bottom=.05)
        rc=REGIME_COLORS.get(reg[d],BLUE)
        fig.suptitle(f'Regime Dashboard — {tk} | {reg[d]} | Day {d}',
                     color=rc,fontsize=14,fontweight='bold',y=.97,fontfamily='monospace')
        dy=np.arange(d+1)
        ax=fig.add_subplot(gs[0,:]); _s(ax)
        for i in range(1,d+1): ax.plot([i-1,i],[c[i-1],c[i]],color=REGIME_COLORS.get(reg[i],BLUE),lw=1.2,alpha=.85)
        ax.set_title(f'{tk} Price — Colored by Regime',color=TEXT,fontsize=10,fontweight='bold')
        ax.set_ylabel('Price',color=DIM,fontsize=8)
        ax.set_xlim(0,N)
        ax=fig.add_subplot(gs[1,0]); _s(ax)
        ax.plot(dy,rv[:d+1],color=YELLOW,lw=1.2)
        ax.axhline(50,color=YELLOW,lw=.7,ls='--',alpha=.6,label='Cautious (50%)')
        ax.axhline(65,color=RED,lw=.7,ls='--',alpha=.6,label='Crisis (65%)')
        ax.set_title('20d Rolling Vol (ann%)',color=TEXT,fontsize=10,fontweight='bold')
        ax.legend(fontsize=7,loc='upper right',facecolor=BG2,edgecolor=GRID_C,labelcolor=TEXT)
        ax.set_xlim(0,N)
        ax=fig.add_subplot(gs[1,1]); _s(ax)
        labels=['BULL','NORMAL','CAUTIOUS','CRISIS','RECOVERY']
        counts=[reg[:d+1].count(l) for l in labels]
        cols_bar=[REGIME_COLORS[l] for l in labels]
        ax.barh(labels,counts,color=cols_bar,height=.6,alpha=.85)
        ax.set_title('Regime Distribution',color=TEXT,fontsize=10,fontweight='bold')
        for i,v in enumerate(counts): ax.text(max(v,0)+1,i,str(v),color=TEXT,fontsize=8,va='center')
        ax=fig.add_subplot(gs[2,0]); _s(ax)
        for i in range(1,d+1): ax.plot([i-1,i],[sp_mult[i-1],sp_mult[i]],color=REGIME_COLORS.get(reg[i],BLUE),lw=1.2,alpha=.8)
        ax.set_title('MM Spread Multiplier',color=TEXT,fontsize=10,fontweight='bold')
        ax.set_ylabel('Multiplier',color=DIM,fontsize=8)
        ax.set_ylim(0.5,3.5); ax.set_xlim(0,N)
        ax=fig.add_subplot(gs[2,1]); _s(ax)
        for i in range(1,d+1): ax.plot([i-1,i],[pos[i-1]*100,pos[i]*100],color=REGIME_COLORS.get(reg[i],BLUE),lw=1.2,alpha=.8)
        ax.axhline(100,color=DIM,lw=.5,ls='--')
        ax.set_title('Base Position Sizing (%)',color=TEXT,fontsize=10,fontweight='bold')
        ax.set_ylabel('Position %',color=DIM,fontsize=8)
        ax.set_ylim(0,140); ax.set_xlim(0,N)
        frames.append(_img(fig,1600,1100)); plt.close(fig)
        if (fi+1)%10==0: print(f'    {fi+1}/{N_FRAMES}')
    _gif(frames,OUT_DIR/'hip3_regime_dashboard.gif',dur=200)

def gen_heatmap(data):
    print('\n[6/7] hip3_vol_spread_heatmap.png ...')
    arb=IVArbitrageEngine(); assets=list(data.keys())
    nd=min(min(len(df) for df in data.values()),600); sm=[]
    for tk in assets:
        c=data[tk]['close'].values[:nd]
        sm.append((arb.compute_hip3_implied_vol(c)-arb.compute_ibkr_atm_iv(c))*100)
    sm=np.array(sm)
    fig,ax=plt.subplots(figsize=(16,6),facecolor=BG); ax.set_facecolor(BG2)
    cm=LinearSegmentedColormap.from_list('s',['#0044ff','#001133','#000000','#331100','#ff4400'])
    v=np.percentile(np.abs(sm),95)
    im=ax.imshow(sm,aspect='auto',cmap=cm,vmin=-v,vmax=v,interpolation='bilinear')
    ax.set_yticks(range(len(assets))); ax.set_yticklabels(assets,fontsize=8,color=TEXT)
    ax.set_xlabel('Day',color=DIM); ax.set_title('IV Spread HIP-3 minus IBKR (%)',color=TEXT,fontsize=11,fontweight='bold')
    ax.tick_params(colors=DIM,labelsize=7)
    for s in ax.spines.values(): s.set_color(GRID_C)
    cb=fig.colorbar(im,ax=ax,fraction=.02,pad=.02); cb.ax.tick_params(colors=TEXT,labelsize=7)
    fig.tight_layout(); fig.savefig(OUT_DIR/'hip3_vol_spread_heatmap.png',dpi=150,facecolor=fig.get_facecolor())
    plt.close(fig); print('  Saved hip3_vol_spread_heatmap.png')

def gen_summary(csv_path):
    print('\n[7/7] hip3_arbitrage_summary.png ...')
    if not csv_path.exists(): print('  No CSV, skip'); return
    df=pd.read_csv(csv_path)
    if df.empty: return
    fig,axes=plt.subplots(2,2,figsize=(16,10),facecolor=BG); fig.subplots_adjust(hspace=.4,wspace=.35)
    a_sorted=df.sort_values('alpha',ascending=True)['asset'].values
    al=df.sort_values('alpha',ascending=True)['alpha'].values*100; x=np.arange(len(a_sorted))
    ax=axes[0,0]; ax.set_facecolor(BG2)
    ax.barh(a_sorted,al,color=[GREEN if v>0 else RED for v in al],height=.6,alpha=.85)
    ax.axvline(0,color=WHITE,lw=.5,alpha=.5); ax.set_title('Alpha %',color=WHITE,fontsize=11,fontweight='bold')
    ax.tick_params(colors=DIM,labelsize=8); [s.set_color(GRID_C) for s in ax.spines.values()]
    ax=axes[0,1]; ax.set_facecolor(BG2)
    ax.barh(x-.15,df.set_index('asset').loc[a_sorted,'sharpe'].values,height=.3,color=CYAN,alpha=.85,label='Strat')
    ax.barh(x+.15,df.set_index('asset').loc[a_sorted,'sharpe_bench'].values,height=.3,color='#666699',alpha=.6,label='Bench')
    ax.set_yticks(x); ax.set_yticklabels(a_sorted,fontsize=8,color=TEXT)
    ax.set_title('Sharpe',color=WHITE,fontsize=11,fontweight='bold')
    ax.legend(fontsize=8,facecolor=BG2,edgecolor=GRID_C,labelcolor=TEXT)
    ax.tick_params(colors=DIM,labelsize=8); [s.set_color(GRID_C) for s in ax.spines.values()]
    ax=axes[1,0]; ax.set_facecolor(BG2)
    ax.barh(x-.15,-df.set_index('asset').loc[a_sorted,'max_dd'].values*100,height=.3,color=YELLOW,alpha=.85,label='Strat')
    ax.barh(x+.15,-df.set_index('asset').loc[a_sorted,'max_dd_bench'].values*100,height=.3,color='#666699',alpha=.6,label='Bench')
    ax.set_yticks(x); ax.set_yticklabels(a_sorted,fontsize=8,color=TEXT)
    ax.set_title('MaxDD %',color=WHITE,fontsize=11,fontweight='bold')
    ax.legend(fontsize=8,facecolor=BG2,edgecolor=GRID_C,labelcolor=TEXT)
    ax.tick_params(colors=DIM,labelsize=8); [s.set_color(GRID_C) for s in ax.spines.values()]
    ax=axes[1,1]; ax.set_facecolor(BG2)
    ax.barh(a_sorted,df.set_index('asset').loc[a_sorted,'calmar'].values,height=.6,color='#cc66ff',alpha=.85)
    ax.set_title('Calmar',color=WHITE,fontsize=11,fontweight='bold')
    ax.tick_params(colors=DIM,labelsize=8); [s.set_color(GRID_C) for s in ax.spines.values()]
    fig.suptitle('HIP-3 IV Arbitrage Summary',fontsize=14,fontweight='bold',color=WHITE,y=.98)
    fig.savefig(OUT_DIR/'hip3_arbitrage_summary.png',dpi=150,facecolor=fig.get_facecolor())
    plt.close(fig); print('  Saved hip3_arbitrage_summary.png')

def main():
    print('='*70); print('  HIP-3 Visualization Generator (Equities/Commodities/ETFs)'); print('='*70)
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    # Use real Hyperliquid HIP-3 candles + funding when cached; else synthetic.
    data = load_real_hip3_data(min_days=100)
    if data:
        print(f'\nLoaded REAL HIP-3 data for {len(data)} assets from Hyperliquid API cache.')
    else:
        print('\nGenerating synthetic HIP-3 data (cache missing) ...')
        data = generate_synthetic_hip3_data(n_assets=25, min_days=100)
    arb=IVArbitrageEngine(); all_bt={}; curves={}
    for tk,df in data.items():
        c,o,h,l=df['close'].values,df['open'].values,df['high'].values,df['low'].values
        f,_=generate_funding_history(c,seed=hash(tk)%2**31)
        bt=rolling_backtest_asset(o,h,l,c,f,arb.compute_hip3_implied_vol(c),arb.compute_ibkr_atm_iv(c),tk)
        all_bt[tk]=bt; curves[tk]=(bt['daily_pnl'][60:],bt['daily_bench'][60:])
    gen_dashboard(data,all_bt); gen_iv(); gen_greeks(); gen_equity(curves); gen_regime(data,all_bt); gen_heatmap(data)
    csv=Path(__file__).resolve().parent.parent/'results'/'hip3_rolling_backtest_results.csv'
    if not csv.exists(): csv=Path(__file__).resolve().parent.parent/'results'/'hip3_backtest_results.csv'
    gen_summary(csv); print(f'\nDone! Output: {OUT_DIR}')

if __name__=='__main__': main()
