"""
DeepReality — Paralel PIN Orkestratörü (Pipeline)
═════════════════════════════════════════════════

PIN Architecture'ın kalbi: her pin bağımsız bir işlem birimidir ve
birbirine bağımlı OLMAYAN tüm pinler PARALEL çalışır. Bir pin başka
bir pinin çıktısına ihtiyaç duyuyorsa (örn. XAI pinleri model
kararlarını görselleştirir), bağımlılık grafiği (DAG) üzerinden
otomatik olarak doğru sıraya konur.

Kullanım:
    pipeline = PinPipeline(max_workers=8)
    pipeline.add_pin(pin_a1)                                  # bağımsız
    pipeline.add_pin(pin_d1, depends_on=["PIN-B1", "PIN-B2"]) # bağımlı
    run = pipeline.run(image_path, on_pin_complete=callback)

    run.results["PIN-A1"]   → standart PIN JSON çıktısı
    run.durations["PIN-A1"] → pin süresi (saniye)
    run.total_time          → toplam duvar saati süresi
    run.sequential_time     → pinlerin toplam süresi (sıralı çalışsaydı)

Bağımlı pinlere geçirilen context:
    {
        "PIN-B1": {...PIN-B1'in tam sonucu...},
        "_pins":  {"PIN-B1": <PinB1Clip instance>}   # model/cache paylaşımı için
    }

Not: Paralellik ThreadPoolExecutor ile sağlanır. PyTorch inference,
NumPy/OpenCV ve I/O ağırlıklı işlemler GIL'i bıraktığı için thread
tabanlı paralellik bu iş yükünde etkilidir; ayrıca modeller process
kopyalamadan (fork maliyeti olmadan) bellekten paylaşılır.
"""

import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field


@dataclass
class PipelineRun:
    """Tek bir görselin pipeline çalışma sonucu."""
    results: dict = field(default_factory=dict)     # pin_id → PIN JSON çıktısı
    durations: dict = field(default_factory=dict)   # pin_id → saniye
    total_time: float = 0.0                          # gerçek (paralel) süre
    sequential_time: float = 0.0                     # sıralı çalışsaydı süre

    @property
    def speedup(self) -> float:
        """Paralel çalışmanın kazandırdığı hız çarpanı."""
        if self.total_time <= 0:
            return 1.0
        return self.sequential_time / self.total_time


@dataclass
class _PinNode:
    pin: object                 # BasePin instance
    depends_on: list[str]       # üst pin_id listesi


class PinPipeline:
    """
    Bağımlılık grafiği (DAG) tabanlı paralel PIN çalıştırıcı.

    - Bağımlılığı olmayan tüm pinler aynı anda başlar.
    - Bir pin, tüm üst pinleri bittiği anda (başkalarını beklemeden) başlar.
    - Bir üst pin hata verse bile alt pin çalıştırılır; hata bilgisi
      context üzerinden alt pine ulaşır (pin kendi kararını verir).
    """

    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self._nodes: dict[str, _PinNode] = {}

    def add_pin(self, pin, depends_on: list[str] | None = None):
        """Pipeline'a bir pin ekler. depends_on: üst pin_id listesi."""
        deps = list(depends_on) if depends_on else []
        self._nodes[pin.pin_id] = _PinNode(pin=pin, depends_on=deps)
        return self

    def _validate(self):
        """Eksik bağımlılık ve döngü kontrolü."""
        for pin_id, node in self._nodes.items():
            for dep in node.depends_on:
                if dep not in self._nodes:
                    raise ValueError(
                        f"{pin_id} pini '{dep}' pinine bağımlı ama "
                        f"'{dep}' pipeline'a eklenmemiş."
                    )
        # Döngü kontrolü (topolojik tüketim)
        resolved: set[str] = set()
        pending = set(self._nodes)
        while pending:
            ready = {
                pid for pid in pending
                if all(d in resolved for d in self._nodes[pid].depends_on)
            }
            if not ready:
                raise ValueError(f"Bağımlılık döngüsü tespit edildi: {pending}")
            resolved |= ready
            pending -= ready

    def run(self, file_path: str, on_pin_complete=None) -> PipelineRun:
        """
        Tüm pinleri tek görsel için paralel çalıştırır.

        on_pin_complete(pin_id, result, duration): her pin bittiğinde
        ana thread'den çağrılır (canlı ilerleme çıktısı için güvenli).
        """
        self._validate()

        run = PipelineRun()
        t0 = time.perf_counter()

        remaining = set(self._nodes)
        running = {}  # Future → pin_id

        def make_job(pin_id: str):
            node = self._nodes[pin_id]

            def job():
                context = {dep: run.results[dep] for dep in node.depends_on}
                if node.depends_on:
                    context["_pins"] = {
                        dep: self._nodes[dep].pin for dep in node.depends_on
                    }
                start = time.perf_counter()
                result = node.pin.run(str(file_path), context=context)
                return result, time.perf_counter() - start

            return job

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            while remaining or running:
                # Bağımlılıkları tamamlanan pinleri hemen başlat
                ready = [
                    pid for pid in remaining
                    if all(dep in run.results
                           for dep in self._nodes[pid].depends_on)
                ]
                for pid in ready:
                    remaining.discard(pid)
                    running[executor.submit(make_job(pid))] = pid

                if not running:
                    break  # _validate döngüyü zaten engeller; güvenlik ağı

                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    pid = running.pop(future)
                    result, duration = future.result()
                    run.results[pid] = result
                    run.durations[pid] = duration
                    if on_pin_complete:
                        on_pin_complete(pid, result, duration)

        run.total_time = time.perf_counter() - t0
        run.sequential_time = sum(run.durations.values())
        return run
